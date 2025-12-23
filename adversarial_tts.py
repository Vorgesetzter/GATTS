# ==== IMPORT FUNCTIONS ====
from __future__ import annotations

import argparse
import logging
import warnings
import datetime
import os
import re
import platform

# ==== Silence Warnings ====
from transformers import logging as hf_logging

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="huggingface_hub.file_download"
)

hf_logging.set_verbosity_error()

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger('phonemizer').setLevel(logging.ERROR)

# === Standard Library ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jiwer
import torch.nn.functional as F
import librosa
from pesq import pesq
import torch
import soundfile as sf
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Wav2Vec2Model, Wav2Vec2Processor
from pymoo.algorithms.moo.nsga2 import NSGA2
from tqdm import tqdm

# === Notification Assistant ===
from dotenv import load_dotenv
import requests

# === From Files ===
# Ensure these files are in the same directory or accessible via PYTHONPATH
try:
    from _helper import addNumbersPattern, adjustInterpolationVector, AttackMode, FitnessObjective, _extend_to_size
    from _pymoo_optimizer import PymooOptimizer
    from _styletts2 import StyleTTS2
    from _asr_model import AutomaticSpeechRecognitionModel
except ImportError as e:
    print(
        "Error importing local modules. Make sure you run this script from the project root directory with 'export PYTHONPATH=Scripts'")
    raise e


def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial TTS Optimization Executable")

    # String parameters
    parser.add_argument("--ground_truth_text", type=str, default="I think the NFL is lame and boring",
                        help="The ground truth text input.")
    parser.add_argument("--target_text", type=str, default="The Seattle Seahawks are the best Team in the world",
                        help="The target text input.")

    # Numeric parameters
    parser.add_argument("--loop_count", type=int, default=1, help="The loop count to use.")
    parser.add_argument("--num_generations", type=int, default=150,
                        help="Number of generations for the optimizer.")
    parser.add_argument("--pop_size", type=int, default=100,
                        help="Population size.")
    parser.add_argument("--iv_scalar", type=float, default=0.5,
                        help="Interpolation vector scalar.")
    parser.add_argument("--size_per_phoneme", type=int, default=1,
                        help="Size per phoneme.")

    # Boolean parameters
    parser.add_argument("--notify", action="store_true",
                        help="If set, sends a WhatsApp notification upon completion.")

    # Enum/Selection parameters
    parser.add_argument("--mode", type=str, default="TARGETED",
                        choices=["TARGETED", "UNTARGETED", "NOISE_UNTARGETED"],
                        help="Attack mode (case sensitive).")

    parser.add_argument("--ACTIVE_OBJECTIVES", nargs="+", type=str,
                        default=["PESQ", "WER_GT"],
                        help="List of active objectives (e.g. PESQ WER_GT UTMOS).")

    parser.add_argument("--thresholds", nargs='*', type=str, default=[],
                        help="Early stopping thresholds. Format: OBJ=Val (e.g. --thresholds PESQ=0.35 WER_GT=0.05)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # ==== Process Arguments ====
    text_gt = args.ground_truth_text
    text_target = args.target_text
    loop_count = args.loop_count
    num_generations = args.num_generations
    pop_size = args.pop_size
    iv_scalar = args.iv_scalar
    size_per_phoneme = args.size_per_phoneme
    notify = args.notify

    # Process Mode Enum
    try:
        mode = AttackMode[args.mode]
    except KeyError:
        print(f"Invalid mode '{args.mode}'. Available modes: {[m.name for m in AttackMode]}")
        return

    # Process Objectives Enum
    ACTIVE_OBJECTIVES = set()
    for obj_name in args.ACTIVE_OBJECTIVES:
        try:
            ACTIVE_OBJECTIVES.add(FitnessObjective[obj_name])
        except KeyError:
            print(f"Warning: '{obj_name}' is not a valid FitnessObjective. Skipping.")

    if not ACTIVE_OBJECTIVES:
        print("Error: No valid ACTIVE_OBJECTIVES selected.")
        return

    # ==== Process Thresholds ====
    THRESHOLDS = {}
    if args.thresholds:
        print("Parsing Thresholds...")
        for t in args.thresholds:
            try:
                # Split "PESQ=0.35" into key and value
                key_str, val_str = t.split("=")

                # Convert string to Enum
                obj_enum = FitnessObjective[key_str]
                val_float = float(val_str)

                THRESHOLDS[obj_enum] = val_float
                print(f"  -> Early Stop if {obj_enum.name} <= {val_float}")
            except Exception as e:
                print(f"  [!] Error parsing threshold '{t}': {e}. Format must be OBJECTIVE=NUMBER")
                return

    print("=== Configuration ===")
    print(f"Mode: {mode.name}")
    print(f"GT Text: {text_gt}")
    print(f"Target Text: {text_target}")
    print(f"Generations: {num_generations}, Pop Size: {pop_size}")
    print(f"Objectives: {[o.name for o in ACTIVE_OBJECTIVES]}")
    print("=====================")

    load_dotenv()
    WHATSAPP_PHONE_NUMBER = os.getenv("WHATSAPP_PHONE_NUMBER")
    WHATSAPP_API_KEY = os.getenv("WHATSAPP_API_KEY")
    text_message = "Iteration finished, check results!"

    # ==== Set Constants ====
    OBJECTIVE_ORDER: list[FitnessObjective] = [
        FitnessObjective.PHONEME_COUNT,
        FitnessObjective.AVG_LOGPROB,
        FitnessObjective.UTMOS,
        FitnessObjective.PPL,
        FitnessObjective.PESQ,
        FitnessObjective.L1,
        FitnessObjective.L2,
        FitnessObjective.WER_TARGET,
        FitnessObjective.SBERT_TARGET,
        FitnessObjective.TEXT_EMB_TARGET,
        FitnessObjective.WER_GT,
        FitnessObjective.SBERT_GT,
        FitnessObjective.TEXT_EMB_GT,
        FitnessObjective.WAV2VEC_SIMILAR,
        FitnessObjective.WAV2VEC_DIFFERENT,
        FitnessObjective.WAV2VEC_ASR,
    ]

    diffusion_steps = 5
    embedding_scale = 1
    subspace_optimization = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    noise = torch.randn(1, 1, 256).to(device)
    random_matrix = np.random.rand(size_per_phoneme, 512)
    random_matrix = torch.from_numpy(random_matrix).to(device).float()

    # ==== Required Models ====

    # 1) Load StyleTTS2
    print("Loading StyleTTS2...")
    tts_model = StyleTTS2()
    tts_model.load_models()
    tts_model.load_checkpoints()
    tts_model.sample_diffusion()

    # 2) Extract Embedding Vectors
    if mode is AttackMode.TARGETED:
        tokens_gt, tokens_target = addNumbersPattern(
            tts_model.preprocessText(text_gt),
            tts_model.preprocessText(text_target),
            [16, 4]
        )
        h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts_model.extract_embeddings(tokens_gt)
        h_text_target, h_bert_raw_target, h_bert_target, _, _ = tts_model.extract_embeddings(tokens_target)
    else:
        tokens_gt = tts_model.preprocessText(text_gt)
        h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts_model.extract_embeddings(tokens_gt)

        h_text_target = torch.randn_like(h_text_gt)
        h_text_target /= h_text_target.norm()
        h_bert_raw_target = torch.randn_like(h_bert_raw_gt)
        h_bert_raw_target /= h_bert_raw_target.norm()
        h_bert_target = torch.randn_like(h_bert_gt)
        h_bert_target /= h_bert_target.norm()

    # 3) Generate Style Vector
    style_vector_acoustic, style_vector_prosodic = tts_model.computeStyleVector(noise, h_bert_raw_gt, embedding_scale, diffusion_steps)

    # 4) Run rest of inference for ground-truth and target
    audio_gt = tts_model.inference_after_interpolation(input_lengths, text_mask, h_bert_gt, h_text_gt, style_vector_acoustic, style_vector_prosodic)
    audio_target = tts_model.inference_after_interpolation(input_lengths, text_mask, h_bert_target, h_text_target, style_vector_acoustic, style_vector_prosodic)

    # 5) Load ASR Model
    print("Loading ASR Model...")
    asr_model = AutomaticSpeechRecognitionModel("tiny", device=device)

    # 6) Load Optimizer
    phoneme_count = input_lengths.detach().cpu().item()
    optimizer = PymooOptimizer(
        bounds=(0, 1),
        algorithm=NSGA2,
        algo_params={"pop_size": pop_size},
        num_objectives=len(ACTIVE_OBJECTIVES),
        solution_shape=(phoneme_count, size_per_phoneme),
    )

    # ==== Conditional Models ====
    # Only load models needed for the active objectives
    if FitnessObjective.TEXT_EMB_TARGET in ACTIVE_OBJECTIVES or FitnessObjective.TEXT_EMB_GT in ACTIVE_OBJECTIVES:
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        embedding_model.eval()
        text_embedding_gt = embedding_model.encode(text_gt, convert_to_tensor=True, normalize_embeddings=True)
        if mode is AttackMode.TARGETED:
            text_embedding_target = embedding_model.encode(text_target, convert_to_tensor=True,
                                                           normalize_embeddings=True)
        elif mode is AttackMode.NOISE_UNTARGETED:
            text_embedding_target = torch.randn_like(text_embedding_gt)
            text_embedding_target /= text_embedding_target.norm()
        elif mode is AttackMode.UNTARGETED:
            text_embedding_target = None

    if FitnessObjective.UTMOS in ACTIVE_OBJECTIVES:
        utmos_model = torch.jit.load(
            hf_hub_download(repo_id="balacoon/utmos", filename="utmos.jit", repo_type="model", local_dir="./"),
            map_location=device
        )
        utmos_model.eval()

    if FitnessObjective.SBERT_GT in ACTIVE_OBJECTIVES or FitnessObjective.SBERT_TARGET in ACTIVE_OBJECTIVES:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        sbert_model.eval()
        s_bert_embedding_gt = sbert_model.encode(text_gt, convert_to_tensor=True, normalize_embeddings=True)
        if mode is AttackMode.TARGETED:
            s_bert_embedding_target = sbert_model.encode(text_target, convert_to_tensor=True, normalize_embeddings=True)
        elif mode is AttackMode.NOISE_UNTARGETED:
            s_bert_embedding_target = torch.randn_like(s_bert_embedding_gt)
            s_bert_embedding_target /= s_bert_embedding_target.norm()
        elif mode is AttackMode.UNTARGETED:
            s_bert_embedding_target = None

    if FitnessObjective.WER_TARGET in ACTIVE_OBJECTIVES or FitnessObjective.WER_GT in ACTIVE_OBJECTIVES:
        wer_transformations = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ])

    if FitnessObjective.PPL in ACTIVE_OBJECTIVES:
        perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        perplexity_model.eval()
        perplexity_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if any(x in ACTIVE_OBJECTIVES for x in [FitnessObjective.WAV2VEC_SIMILAR, FitnessObjective.WAV2VEC_DIFFERENT, FitnessObjective.WAV2VEC_ASR]):
        wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
        wav2vec_model.eval()
        with torch.no_grad():
            wav2vec_embedding_gt = torch.mean(wav2vec_model(
                **wav2vec_processor(audio_gt, return_tensors="pt", sampling_rate=16000).to(device)).last_hidden_state,
                                              dim=1)
            if mode is AttackMode.TARGETED:
                wav2vec_embedding_target = torch.mean(wav2vec_model(
                    **wav2vec_processor(audio_target, return_tensors="pt", sampling_rate=16000).to(
                        device)).last_hidden_state, dim=1)
            elif mode is AttackMode.NOISE_UNTARGETED:
                wav2vec_embedding_target = torch.randn_like(wav2vec_embedding_gt)
                wav2vec_embedding_target /= wav2vec_embedding_target.norm()
            elif mode is AttackMode.UNTARGETED:
                wav2vec_embedding_target = None

    print(f"Starting Optimization Loop...")

    for iteration in tqdm(range(loop_count), desc="Total Progress"):

        # ==== Main Optimization Loop ====
        fitness_history = []
        mean_model = []
        stop_optimization = False

        progress_bar = tqdm(range(num_generations), desc=f"Current Generation {iteration+1}", leave=False)
        gen = -1

        for gen in progress_bar:
            gen_scores: dict[FitnessObjective, list[float]] = {obj: [] for obj in ACTIVE_OBJECTIVES}
            population_vectors = optimizer.get_x_current()

            for j, interpolation_vector_np in enumerate(population_vectors):
                IV = torch.from_numpy(interpolation_vector_np).to(device).float()
                interpolation_vector = adjustInterpolationVector(IV, random_matrix, subspace_optimization)

                # Initialize dictionary using Enum keys type hint
                current_ind_scores: dict[FitnessObjective, float] = {}


                # Interpolate Values depending on AttackMode
                if mode is AttackMode.NOISE_UNTARGETED or mode is AttackMode.TARGETED:
                    h_text_mixed = (1.0 - interpolation_vector) * h_text_gt + interpolation_vector * h_text_target
                else:
                    if h_text_gt.shape != interpolation_vector.shape:
                        raise ValueError(
                            "AttackMode.UNTARGETED requires h_text_gt and interpolation_vector to be of same shape.")
                    h_text_mixed = h_text_gt + iv_scalar * interpolation_vector

                h_bert_mixed = h_bert_gt

                audio_mixed = tts_model.inference_after_interpolation(
                    input_lengths, text_mask, h_bert_mixed, h_text_mixed, style_vector_acoustic, style_vector_prosodic
                )

                # ASR Analysis
                asr_result, asr_logprob = asr_model.analyzeAudio(audio_mixed)
                asr_text = asr_result["text"]
                # Force English characters (A-Z, a-z, spaces)
                clean_text = re.sub(r'[^a-zA-Z\s]', '', asr_text).strip()

                # Handle garbage text
                if len(clean_text) < 2:
                    val = 10.0
                    for obj in ACTIVE_OBJECTIVES:
                        gen_scores[obj].append(val)
                        current_ind_scores[obj] = val
                    record = {"Generation": gen, "Individual_ID": j}
                    record.update(current_ind_scores)
                    fitness_history.append(record)
                    continue

                asr_text = clean_text

                # ==== Increase Naturalness ====
                if FitnessObjective.PHONEME_COUNT in ACTIVE_OBJECTIVES:
                    tokens_asr = tts_model.preprocessText(asr_text)
                    n_asr = int(tokens_asr.shape[-1])
                    n_gt = int(input_lengths.item())

                    if n_asr == 0 or n_asr > n_gt * 2:
                        # Assign penalty to all active objectives if ASR failed completely
                        val = 1.0
                    else:
                        error = abs(n_asr - n_gt) / max(1, n_gt)
                        val = float(min(1.0, error * error))

                    gen_scores[FitnessObjective.PHONEME_COUNT].append(val)
                    current_ind_scores[FitnessObjective.PHONEME_COUNT] = val

                if FitnessObjective.AVG_LOGPROB in ACTIVE_OBJECTIVES:

                    # asr_logprob = mean(log(probability_token)) [average log of token_probability]
                    # Values = usually (-3, 0), rarely < -3.0
                    # -3 ~ log(0.05) = 5% Probability of token, 0 = 100% Probability of token

                    val = - float((asr_logprob / 3.0))

                    gen_scores[FitnessObjective.AVG_LOGPROB].append(val)
                    current_ind_scores[FitnessObjective.AVG_LOGPROB] = val

                if FitnessObjective.UTMOS in ACTIVE_OBJECTIVES:

                    # predicted_mos = utmos_model(audio).item()
                    # Values: [1, 5]
                    # 1 = bad audio, 5 = perfect audio

                    audio_mos = torch.as_tensor(audio_mixed, dtype=torch.float32, device=device).unsqueeze(0)
                    predicted_mos = utmos_model(audio_mos).item()

                    val = (predicted_mos - 1.0) / 4.0
                    val = - val + 1

                    gen_scores[FitnessObjective.UTMOS].append(val)
                    current_ind_scores[FitnessObjective.UTMOS] = val

                if FitnessObjective.PPL in ACTIVE_OBJECTIVES:

                    # ppl_naturalness = GPT-2 perplexity: the more surprised the model is by the text,
                    # Values: usually (0, 1)
                    # 0.0 = very unnatural sentence (rare, strange, or ungrammatical), 1.0 = very natural, fluent sentence (likely to be common human language)

                    min_loss = 1.0
                    max_loss = 10.0

                    ppl_tokens = perplexity_tokenizer(asr_text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = perplexity_model(**ppl_tokens, labels=ppl_tokens["input_ids"])
                        loss = outputs.loss

                    loss_val = float(loss.item())
                    loss_clamped = max(min_loss, min(loss_val, max_loss))
                    ppl_naturalness = 1.0 - (loss_clamped - min_loss) / (max_loss - min_loss)

                    val = float(ppl_naturalness)
                    val = - val + 1.0

                    gen_scores[FitnessObjective.PPL].append(val)
                    current_ind_scores[FitnessObjective.PPL] = val

                if FitnessObjective.PESQ in ACTIVE_OBJECTIVES:

                    # score_pesq
                    # Values: [-0.5, 4.5]
                    # -0.5 absolute floor / unintelligible, 4.5 = Perfect audio

                    audio_gt_16khz = librosa.resample(audio_gt, orig_sr=24000, target_sr=16000)
                    audio_mixed_16khz = librosa.resample(audio_mixed, orig_sr=24000, target_sr=16000)

                    score_pesq = pesq(16000, audio_gt_16khz, audio_mixed_16khz, 'wb')

                    val = score_pesq + 0.5
                    val /= 5.0
                    val = - val + 1.0

                    gen_scores[FitnessObjective.PESQ].append(val)
                    current_ind_scores[FitnessObjective.PESQ] = val

                # ==== Interpolation Vector Restrictions ====
                if FitnessObjective.L1 in ACTIVE_OBJECTIVES:

                    # L1 = mean(|IV|) [Average value of interpolation vector]
                    # Values: (0,1)
                    # 0 = only GT, 1 = only Target

                    val = float(interpolation_vector.abs().mean().item())

                    gen_scores[FitnessObjective.L1].append(val)
                    current_ind_scores[FitnessObjective.L1] = val

                if FitnessObjective.L2 in ACTIVE_OBJECTIVES:

                    # L2 = sqrt(mean(IV²)) [Average, but punishes larger numbers more]
                    # Values: (0,1)
                    # 0 = only GT, 1 = only Target

                    val = float((interpolation_vector ** 2).mean().sqrt().item())

                    gen_scores[FitnessObjective.L2].append(val)
                    current_ind_scores[FitnessObjective.L2] = val

                # ==== Optimize Text Towards Target ====
                if FitnessObjective.WER_TARGET in ACTIVE_OBJECTIVES:

                    # wer = (Substitutions + Deletions + Insertions) / Number_of_reference_words
                    # Values: usually (0, 1), rarely > 1
                    # 0 = perfect, 1 = 100% of words wrong

                    wer = jiwer.wer(
                        text_target,
                        asr_text,
                        reference_transform=wer_transformations,
                        hypothesis_transform=wer_transformations,
                    )

                    val = float(wer)

                    gen_scores[FitnessObjective.WER_TARGET].append(val)
                    current_ind_scores[FitnessObjective.WER_TARGET] = val

                if FitnessObjective.SBERT_TARGET in ACTIVE_OBJECTIVES:

                    # sbert_target = cos_sim(emb_target, emb_asr)
                    # Values: [-1, 1]
                    # -1 = ASR very different to Target, 1 = ASR same as Target

                    if mode is AttackMode.UNTARGETED:
                        raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.SBERT_TARGET")

                    sbert_target = util.cos_sim(
                        s_bert_embedding_target,
                        sbert_model.encode(asr_text, convert_to_tensor=True,normalize_embeddings=True)
                    ).item()

                    val = (sbert_target + 1) / 2.0
                    val = - val + 1
                    val = float(val)

                    gen_scores[FitnessObjective.SBERT_TARGET].append(val)
                    current_ind_scores[FitnessObjective.SBERT_TARGET] = val

                if FitnessObjective.TEXT_EMB_TARGET in ACTIVE_OBJECTIVES:

                    # text_dist_target = cos_sim(emb_target, emb_asr)
                    # Values: [-1, 1]
                    # -1 = ASR very different to Target, 1 = ASR same as Target

                    if mode is AttackMode.UNTARGETED:
                        raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.TEXT_EMB_TARGET")

                    text_dist_target = F.cosine_similarity(
                        text_embedding_target,
                        embedding_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True),
                        dim=0
                    ).item()
                    val = (text_dist_target + 1) / 2.0
                    val = - val + 1
                    val = float(val)

                    gen_scores[FitnessObjective.TEXT_EMB_TARGET].append(val)
                    current_ind_scores[FitnessObjective.TEXT_EMB_TARGET] = val

                # ==== Optimize Text Away From Ground-Truth ====
                if FitnessObjective.WER_GT in ACTIVE_OBJECTIVES:
                    # wer = (Substitutions + Deletions + Insertions) / Number_of_reference_words
                    # Values: usually (0, 1), rarely > 1
                    # 0 = perfect, 1 = 100% of words wrong

                    wer = jiwer.wer(
                        text_gt,
                        asr_text,
                        reference_transform=wer_transformations,
                        hypothesis_transform=wer_transformations,
                    )
                    val = float(wer)
                    val = -val + 1.0

                    gen_scores[FitnessObjective.WER_GT].append(val)
                    current_ind_scores[FitnessObjective.WER_GT] = val

                if FitnessObjective.SBERT_GT in ACTIVE_OBJECTIVES:
                    # sbert_gt = cos_sim(emb_gt, emb_asr)
                    # Values: [-1, 1]
                    # -1 = ASR very different to GT, 1 = ASR same as GT

                    sbert_gt = util.cos_sim(
                        s_bert_embedding_gt,
                        sbert_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True)
                    ).item()
                    val = (sbert_gt + 1) / 2.0
                    val = float(val)

                    gen_scores[FitnessObjective.SBERT_GT].append(val)
                    current_ind_scores[FitnessObjective.SBERT_GT] = val

                if FitnessObjective.TEXT_EMB_GT in ACTIVE_OBJECTIVES:
                    # text_dist_gt = cos_sim(emb_gt, emb_asr)
                    # Values: [-1, 1]
                    # -1 = ASR very different to GT, 1 = ASR same as GT

                    text_dist_gt = F.cosine_similarity(
                        text_embedding_gt,
                        embedding_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True),
                        dim=0
                    ).item()
                    val = (text_dist_gt + 1) / 2.0
                    val = float(val)

                    gen_scores[FitnessObjective.TEXT_EMB_GT].append(val)
                    current_ind_scores[FitnessObjective.TEXT_EMB_GT] = val

                # ==== Optimize Audio Similarity ====
                if FitnessObjective.WAV2VEC_SIMILAR in ACTIVE_OBJECTIVES:
                    # wav2vec_gt = cos_sim(emb_gt, emb_asr)
                    # Values = [-1, 1]
                    # -1 = ASR very different to GT, 1 = ASR same as GT

                    with torch.no_grad():
                        wav2vec_embedding_mixed = torch.mean(
                            wav2vec_model(
                                **wav2vec_processor(
                                    audio_mixed, return_tensors="pt", sampling_rate=16000
                                ).to(device)
                            ).last_hidden_state,
                            dim=1
                        )

                    wav2vec_gt = F.cosine_similarity(wav2vec_embedding_gt, wav2vec_embedding_mixed).item()
                    val = (wav2vec_gt + 1) / 2.0
                    val = - val + 1
                    val = float(val)

                    gen_scores[FitnessObjective.WAV2VEC_SIMILAR].append(val)
                    current_ind_scores[FitnessObjective.WAV2VEC_SIMILAR] = val

                if FitnessObjective.WAV2VEC_DIFFERENT in ACTIVE_OBJECTIVES:

                    # wav2vec_target = cos_sim(emb_target, emb_asr)
                    # Values = [-1, 1]
                    # -1 = ASR very different to Target, 1 = ASR same as Target

                    if mode is AttackMode.UNTARGETED:
                        raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.WAV2VEC_DIFFERENT")

                    with torch.no_grad():
                        wav2vec_embedding_mixed = torch.mean(
                            wav2vec_model(
                                **wav2vec_processor(
                                    audio_mixed, return_tensors="pt", sampling_rate=16000
                                ).to(device)
                            ).last_hidden_state,
                            dim=1
                        )

                    wav2vec_sim = F.cosine_similarity(wav2vec_embedding_gt, wav2vec_embedding_mixed).item()
                    val = (wav2vec_sim + 1) / 2.0
                    val = float(val)

                    gen_scores[FitnessObjective.WAV2VEC_DIFFERENT].append(val)
                    current_ind_scores[FitnessObjective.WAV2VEC_DIFFERENT] = val

                if FitnessObjective.WAV2VEC_ASR in ACTIVE_OBJECTIVES:
                    if mode is AttackMode.UNTARGETED:
                        raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.WAV2VEC_ASR")

                    audio_asr = tts_model.inference(asr_text, noise)

                    with torch.no_grad():
                        wav2vec_embedding_asr = torch.mean(
                            wav2vec_model(
                                **wav2vec_processor(
                                    audio_asr, return_tensors="pt", sampling_rate=16000
                                ).to(device)
                            ).last_hidden_state,
                            dim=1
                        )

                        wav2vec_embedding_mixed = torch.mean(
                            wav2vec_model(
                                **wav2vec_processor(
                                    audio_mixed, return_tensors="pt", sampling_rate=16000
                                ).to(device)
                            ).last_hidden_state,
                            dim=1
                        )

                    wav2vec_asr = F.cosine_similarity(wav2vec_embedding_asr, wav2vec_embedding_mixed).item()
                    val = (wav2vec_asr + 1) / 2.0
                    val = - val + 1
                    val = float(val)

                    gen_scores[FitnessObjective.WAV2VEC_ASR].append(val)
                    current_ind_scores[FitnessObjective.WAV2VEC_ASR] = val

                # ==== EARLY STOPPING CHECK ====
                # Only run this logic if the user actually provided thresholds via terminal
                if THRESHOLDS:
                    meets_all_criteria = True

                    for obj in ACTIVE_OBJECTIVES:
                        # We only care about objectives that HAVE a threshold set
                        if obj in THRESHOLDS:
                            current_fitness = current_ind_scores[obj]
                            target_fitness = THRESHOLDS[obj]

                            # Optimization Goal: MINIMIZE Fitness.
                            # We fail if: current_fitness > target_fitness
                            if current_fitness > target_fitness:
                                meets_all_criteria = False
                                break

                                # If we survived the loop above, this individual passed all checks
                    if meets_all_criteria:
                        stop_optimization = True

                # Store record
                record = {"Generation": gen, "Individual_ID": j}
                record.update(current_ind_scores)
                fitness_history.append(record)

            # 3. Calculate per-generation means
            gen_mean: dict[str, float] = {"Generation": gen}
            fitness_arrays_for_optimizer: list[np.ndarray] = []

            for obj in OBJECTIVE_ORDER:
                if obj not in ACTIVE_OBJECTIVES:
                    continue

                arr = np.array(gen_scores[obj], dtype=float)

                gen_mean[f"{obj.name}_Mean"] = float(np.mean(arr))
                fitness_arrays_for_optimizer.append(arr)

            mean_model.append(gen_mean)

            # 4. Update Optimizer
            optimizer.assign_fitness(fitness_arrays_for_optimizer)
            optimizer.update()

            if stop_optimization:
                print(f"\n[!] Early Stopping Triggered at Generation {gen + 1} (Thresholds met).")
                break

        # ==== Afterwork (Inference + Logging) ====
        # 1. Bundle Runtime Data (Things that changed during the loop)
        run_context = {
            "fitness_history": fitness_history,
            "mean_model": mean_model,
            "progress_bar": progress_bar,  # Pass the actual tqdm object!
            "current_gen": gen,
            "stop_optimization": stop_optimization,
            "active_objectives": ACTIVE_OBJECTIVES,
            "objective_order": OBJECTIVE_ORDER,
            "thresholds": THRESHOLDS if 'THRESHOLDS' in locals() else None
        }

        # 2. Bundle Embedding Data (Things used for inference)
        embedding_data = {
            "h_text_gt": h_text_gt,
            "h_text_target": h_text_target,
            "h_bert_gt": h_bert_gt,
            "h_bert_target": h_bert_target,
            "input_lengths": input_lengths,
            "text_mask": text_mask,
            "style_vector_acoustic": style_vector_acoustic,
            "style_vector_prosodic": style_vector_prosodic,
            "random_matrix": random_matrix,
            "phoneme_count": phoneme_count
        }

        # 3. The Single Call
        finalize_run(optimizer, tts_model, asr_model, args, run_context, embedding_data, device)


if __name__ == "__main__":
    main()