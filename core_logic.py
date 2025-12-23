import torch
import numpy as np
import re
import librosa
from pesq import pesq
import jiwer
import torch.nn.functional as F
from sentence_transformers import util
from tqdm import tqdm
from _helper import adjustInterpolationVector, AttackMode, FitnessObjective


def run_optimization_loop(optimizer, loop_count, num_generations, context):
    """
    Main entry point for the optimization process.
    """
    device = context['device']
    active_objectives = context['active_objectives']

    # Trackers
    fitness_history = []
    mean_model = []
    stop_optimization = False

    # Outer Loop (Iterations) is handled by the caller or tqdm here
    # If loop_count is handled inside this function:
    # for iteration in tqdm(range(loop_count), desc="Total Progress"):

    # We assume this function runs ONE full optimization cycle (loop_count=1 logic)
    # or you can wrap it. Let's stick to your structure where this IS the loop.

    gen = -1
    progress_bar = tqdm(range(num_generations), desc="Optimization Progress", leave=False)

    for gen in progress_bar:
        gen_scores = {obj: [] for obj in active_objectives}
        population_vectors = optimizer.get_x_current()

        # --- Batch Evaluation ---
        # We loop through the population here
        for j, interpolation_vector_np in enumerate(population_vectors):

            # 1. Evaluate Single Candidate
            scores, is_valid = _evaluate_candidate(interpolation_vector_np, context)

            # 2. Store Scores
            if not is_valid:
                # Penalty for failure (e.g. garbage text)
                penalty_val = 10.0
                for obj in active_objectives:
                    gen_scores[obj].append(penalty_val)
                    scores[obj] = penalty_val  # Update record for history
            else:
                for obj in active_objectives:
                    gen_scores[obj].append(scores[obj])

            # 3. History logging
            record = {"Generation": gen, "Individual_ID": j}
            record.update(scores)
            fitness_history.append(record)

            # 4. Check Individual Early Stopping (Thresholds)
            if context.get('thresholds') and is_valid:
                if _check_thresholds(scores, context['thresholds']):
                    stop_optimization = True

        # --- End of Generation Updates ---

        # Calculate Means
        gen_mean = {"Generation": gen}
        fitness_arrays = []

        for obj in context['objective_order']:
            if obj in active_objectives:
                arr = np.array(gen_scores[obj], dtype=float)
                gen_mean[f"{obj.name}_Mean"] = float(np.mean(arr))
                fitness_arrays.append(arr)

        mean_model.append(gen_mean)

        # Update Optimizer
        optimizer.assign_fitness(fitness_arrays)
        optimizer.update()

        if stop_optimization:
            print(f"\n[!] Early Stopping Triggered at Generation {gen + 1}.")
            break

    return fitness_history, mean_model, progress_bar, stop_optimization, gen


def _evaluate_candidate(iv_numpy, ctx):
    """
    Performs inference and calculating scoring for a SINGLE candidate.
    Returns: (scores_dict, is_valid_boolean)
    """
    device = ctx['device']
    mode = ctx['mode']
    active_objectives = ctx['active_objectives']

    # 1. Prepare Vector
    IV = torch.from_numpy(iv_numpy).to(device).float()
    interpolation_vector = adjustInterpolationVector(IV, ctx['random_matrix'], ctx['subspace_optimization'])

    # 2. Mix Embeddings
    if mode is AttackMode.NOISE_UNTARGETED or mode is AttackMode.TARGETED:
        h_text_mixed = (1.0 - interpolation_vector) * ctx['h_text_gt'] + interpolation_vector * ctx['h_text_target']
    else:
        # Untargeted logic
        h_text_mixed = ctx['h_text_gt'] + ctx['iv_scalar'] * interpolation_vector

    # 3. Inference
    audio_mixed = ctx['tts_model'].inference_after_interpolation(
        ctx['input_lengths'], ctx['text_mask'], ctx['h_bert_gt'], h_text_mixed,
        ctx['style_vector_acoustic'], ctx['style_vector_prosodic']
    )

    # 4. ASR
    asr_result, asr_logprob = ctx['asr_model'].analyzeAudio(audio_mixed)
    asr_text_raw = asr_result["text"]
    clean_text = re.sub(r'[^a-zA-Z\s]', '', asr_text_raw).strip()

    # Garbage Check
    if len(clean_text) < 2:
        return {}, False

    # 5. Calculate Objectives
    scores = {}

    # --- Text Metrics ---
    if FitnessObjective.PHONEME_COUNT in active_objectives:
        tokens_asr = ctx['tts_model'].preprocessText(clean_text)
        n_asr = int(tokens_asr.shape[-1])
        n_gt = int(ctx['input_lengths'].item())
        if n_asr == 0 or n_asr > n_gt * 2:
            scores[FitnessObjective.PHONEME_COUNT] = 1.0
        else:
            error = abs(n_asr - n_gt) / max(1, n_gt)
            scores[FitnessObjective.PHONEME_COUNT] = float(min(1.0, error * error))

    if FitnessObjective.AVG_LOGPROB in active_objectives:
        scores[FitnessObjective.AVG_LOGPROB] = -float((asr_logprob / 3.0))

    if FitnessObjective.WER_TARGET in active_objectives:
        wer = jiwer.wer(ctx['text_target'], clean_text, reference_transform=ctx['wer_trans'],
                        hypothesis_transform=ctx['wer_trans'])
        scores[FitnessObjective.WER_TARGET] = float(wer)

    if FitnessObjective.WER_GT in active_objectives:
        wer = jiwer.wer(ctx['text_gt'], clean_text, reference_transform=ctx['wer_trans'],
                        hypothesis_transform=ctx['wer_trans'])
        scores[FitnessObjective.WER_GT] = -float(wer) + 1.0

    # --- Audio Metrics ---
    if FitnessObjective.PESQ in active_objectives:
        # Resample on CPU for PESQ
        aud_gt_16 = librosa.resample(ctx['audio_gt'], orig_sr=24000, target_sr=16000)
        aud_mix_16 = librosa.resample(audio_mixed, orig_sr=24000, target_sr=16000)
        try:
            p_score = pesq(16000, aud_gt_16, aud_mix_16, 'wb')
            val = (p_score + 0.5) / 5.0
            scores[FitnessObjective.PESQ] = -val + 1.0
        except:
            scores[FitnessObjective.PESQ] = 1.0  # Penalty if PESQ fails

    if FitnessObjective.UTMOS in active_objectives:
        audio_mos = torch.as_tensor(audio_mixed, dtype=torch.float32, device=device).unsqueeze(0)
        predicted_mos = ctx['utmos_model'](audio_mos).item()
        scores[FitnessObjective.UTMOS] = -((predicted_mos - 1.0) / 4.0) + 1.0

    # --- Embedding Metrics ---
    # (Simplified example: Add your SBERT/Wav2Vec blocks here following the same pattern)
    if FitnessObjective.L2 in active_objectives:
        scores[FitnessObjective.L2] = float((interpolation_vector ** 2).mean().sqrt().item())

    # ... Add the rest of your objective blocks here ...
    # Note: Access models like ctx['sbert_model'] or ctx['wav2vec_model']

    return scores, True


def _check_thresholds(current_scores, thresholds):
    """Returns True if ALL thresholds are passed."""
    for obj, thresh in thresholds.items():
        if obj in current_scores:
            # We minimize fitness. Pass if current <= threshold
            if current_scores[obj] > thresh:
                return False
    return True