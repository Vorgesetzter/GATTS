"""
Transcribe all ground_truth.wav files using Whisper and compare to the
ground_truth_text stored in run_summary.json.

Usage:
    conda run -n styletts2 python Scripts/Analysis/verify_gt_transcriptions.py
    conda run -n styletts2 python Scripts/Analysis/verify_gt_transcriptions.py --tts_root outputs/results/TTS
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import re
import string
import torch
import torchaudio
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from Models.whisper import Whisper

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

_STOPWORDS = set(stopwords.words('english'))
_LEMMATIZER = WordNetLemmatizer()


def normalize(text: str) -> str:
    """Lowercase and strip punctuation for loose comparison."""
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()


def set_overlap(gt_text: str, asr_text: str) -> float:
    """
    Fraction of GT content-word lemmas that survived in the ASR output.
    1.0 = all content words preserved (GT audio is perfectly intelligible).
    Uses the same pipeline as SetOverlapObjective.
    """
    def content_lemmas(text: str) -> set:
        clean = re.sub(r'[^\w\s]', '', text.lower())
        words = set(clean.split()) - _STOPWORDS
        result = set()
        for w in words:
            for pos in ('a', 'v', 'n', 'r'):
                lemma = _LEMMATIZER.lemmatize(w, pos=pos)
                if lemma != w:
                    result.add(lemma)
                    break
            else:
                result.add(w)
        return result

    gt_lemmas = content_lemmas(gt_text)
    if not gt_lemmas:
        return 1.0
    asr_lemmas = content_lemmas(asr_text)
    return len(gt_lemmas & asr_lemmas) / len(gt_lemmas)


def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Mix to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # (1, samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts_root", default="outputs/results/TTS")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    asr = Whisper(device=device)

    # Collect one GT file per sentence (run_0, falling back to any available run)
    sentence_dirs = sorted([
        d for d in os.listdir(args.tts_root)
        if d.startswith("sentence_") and os.path.isdir(os.path.join(args.tts_root, d))
    ])

    results = []
    for sent_dir in sentence_dirs:
        sent_path = os.path.join(args.tts_root, sent_dir)
        run_dirs = sorted(os.listdir(sent_path))

        gt_wav = None
        gt_text = None
        for run_dir in run_dirs:
            run_path = os.path.join(sent_path, run_dir)
            wav_path = os.path.join(run_path, "ground_truth.wav")
            summary_path = os.path.join(run_path, "run_summary.json")
            if os.path.exists(wav_path) and os.path.exists(summary_path):
                gt_wav = wav_path
                with open(summary_path) as f:
                    gt_text = json.load(f)["text_data"]["ground_truth_text"]
                break

        if gt_wav is None:
            print(f"[SKIP] {sent_dir}: no ground_truth.wav found")
            continue

        audio = load_audio(gt_wav).to(device)
        transcription, _ = asr.inference(audio)
        transcription = transcription[0]

        match = normalize(transcription) == normalize(gt_text)
        overlap = set_overlap(gt_text, transcription)
        results.append({
            "sentence": sent_dir,
            "ground_truth": gt_text,
            "transcription": transcription,
            "match": match,
            "set_overlap": overlap,
        })
        status = "OK" if match else "DIFF"
        print(f"[{status}] {sent_dir}  (set_overlap={overlap:.3f})")
        print(f"  GT:  {gt_text}")
        print(f"  ASR: {transcription}")

    n_total = len(results)
    n_match = sum(r["match"] for r in results)
    avg_overlap = sum(r["set_overlap"] for r in results) / n_total if n_total else 0
    print(f"\n{'='*60}")
    print(f"Exact match:       {n_match}/{n_total} ({n_match/n_total:.1%})")
    print(f"Avg set_overlap:   {avg_overlap:.3f}  (1.0 = all content words preserved)")
    print(f"\nMismatches (sorted by set_overlap ascending):")
    for r in sorted((r for r in results if not r["match"]), key=lambda x: x["set_overlap"]):
        print(f"  {r['sentence']}  overlap={r['set_overlap']:.3f}")
        print(f"    GT:  {r['ground_truth']}")
        print(f"    ASR: {r['transcription']}")


if __name__ == "__main__":
    main()
