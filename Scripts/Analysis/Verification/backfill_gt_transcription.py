"""
Backfill gt_transcription into existing run_summary.json files.

Transcribes the ground_truth.wav in each run folder using Whisper and writes
the result into text_data.gt_transcription if not already present.

Usage:
    conda run -n styletts2 python Scripts/Analysis/backfill_gt_transcription.py
    conda run -n styletts2 python Scripts/Analysis/backfill_gt_transcription.py --tts_root outputs/results/TTS --waveform_root outputs/results/Waveform
    conda run -n styletts2 python Scripts/Analysis/backfill_gt_transcription.py --overwrite
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torchaudio

from Models.whisper import Whisper


def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # (1, samples)


def collect_run_dirs(root: str) -> list[str]:
    runs = []
    if not os.path.isdir(root):
        return runs
    for sent_dir in sorted(os.listdir(root)):
        sent_path = os.path.join(root, sent_dir)
        if not (sent_dir.startswith("sentence_") and os.path.isdir(sent_path)):
            continue
        for run_dir in sorted(os.listdir(sent_path)):
            run_path = os.path.join(sent_path, run_dir)
            if os.path.isdir(run_path):
                runs.append(run_path)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts_root",      default="outputs/results/TTS")
    parser.add_argument("--waveform_root", default="outputs/results/Waveform")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite gt_transcription even if already present")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    asr = Whisper(device=device)

    roots = []
    if os.path.isdir(args.tts_root):
        roots.append(("TTS", args.tts_root))
    if os.path.isdir(args.waveform_root):
        roots.append(("Waveform", args.waveform_root))

    updated = skipped = missing = 0

    for label, root in roots:
        run_dirs = collect_run_dirs(root)
        print(f"\n[{label}] {len(run_dirs)} run dirs under {root}")

        for run_path in run_dirs:
            wav_path     = os.path.join(run_path, "ground_truth.wav")
            summary_path = os.path.join(run_path, "run_summary.json")

            if not os.path.exists(wav_path) or not os.path.exists(summary_path):
                missing += 1
                continue

            with open(summary_path) as f:
                summary = json.load(f)

            already_has = "gt_transcription" in summary.get("text_data", {})
            if already_has and not args.overwrite:
                skipped += 1
                continue

            audio = load_audio(wav_path).to(device)
            transcription, _ = asr.inference(audio)
            transcription = transcription[0]

            summary.setdefault("text_data", {})["gt_transcription"] = transcription

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            updated += 1
            print(f"  [updated] {os.path.relpath(run_path)}  →  \"{transcription}\"")

    print(f"\nDone. updated={updated}, skipped={skipped}, missing={missing}")


if __name__ == "__main__":
    main()
