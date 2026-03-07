"""
Select audio samples for MOS survey.

Design:
  - 10 "both_success" sentences: both methods have ≥1 successful run → pick a successful run
  - 10 "both_fail"    sentences: both methods have ≥1 failed run    → pick a failed run
  - Both groups are disjoint (fail pool sampled first; success pool drawn from remaining)
  - Per sentence: GT + TTS_run + Waveform_run → 3 clips each
  - Total: 60 clips

Survey structure (per participant):
  - Shuffle the 20 sentence groups, pick random version (GT/TTS/Waveform) per group
  - First 10 → Part 1: listen + transcribe + MOS rating
  - Next  10 → Part 2: read transcription + guess original sentence

Output:
  <output_dir>/GT/          gt_<sentence_id:03d>.wav
  <output_dir>/TTS/         tts_<sentence_id:03d>.wav
  <output_dir>/Waveform/    waveform_<sentence_id:03d>.wav
  <output_dir>/manifest.json
  <output_dir>/sentence_groups.js

Usage:
  python Scripts/Analysis/select_survey_samples.py
  python Scripts/Analysis/select_survey_samples.py --seed 99 --n_success 10 --n_failed 10
"""

import argparse
import json
import os
import shutil

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_df(csv_path: str, pesq_col: str, overlap_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={pesq_col: "pesq", overlap_col: "set_overlap"})
    df["sentence_id"] = df["sentence_id"].astype(int)
    df["run_id"] = df["run_id"].astype(int)
    return df


def run_folder(root: str, sentence_id: int, run_id: int) -> str:
    return os.path.join(root, f"sentence_{sentence_id:03d}", f"run_{run_id}")


def random_success_run(df: pd.DataFrame, sentence_id: int, rng: np.random.Generator) -> pd.Series:
    rows = df[(df["sentence_id"] == sentence_id) & (df["success"] == True)]
    return rows.iloc[rng.integers(len(rows))]


def random_failed_run(df: pd.DataFrame, sentence_id: int, rng: np.random.Generator) -> pd.Series:
    rows = df[(df["sentence_id"] == sentence_id) & (df["success"] == False)]
    return rows.iloc[rng.integers(len(rows))]


def copy_clip(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        print(f"  [WARN] Missing: {src}")
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "item"):
            return o.item()
        return super().default(o)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts_csv",       default="outputs/results/TTS/all_results.csv")
    parser.add_argument("--waveform_csv",  default="outputs/analysis/Waveform/all_results.csv")
    parser.add_argument("--tts_root",      default="outputs/results/TTS")
    parser.add_argument("--waveform_root", default="outputs/results/Waveform")
    parser.add_argument("--output_dir",    default="survey")
    parser.add_argument("--n_success",     type=int, default=10,
                        help="Sentences where both methods have >=1 successful run")
    parser.add_argument("--n_failed",      type=int, default=10,
                        help="Sentences where both methods have >=1 failed run")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Clean output directory
    # ------------------------------------------------------------------
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        print(f"Cleared existing output dir: {args.output_dir}")
    os.makedirs(args.output_dir)

    # ------------------------------------------------------------------
    # Load & classify
    # ------------------------------------------------------------------
    tts = load_df(args.tts_csv,      pesq_col="score_PESQ", overlap_col="score_SET_OVERLAP")
    wav = load_df(args.waveform_csv, pesq_col="pesq",        overlap_col="set_overlap")

    shared_ids = sorted(set(tts["sentence_id"].unique()) & set(wav["sentence_id"].unique()))
    tts = tts[tts["sentence_id"].isin(shared_ids)]
    wav = wav[wav["sentence_id"].isin(shared_ids)]

    tts_any_success = tts.groupby("sentence_id")["success"].any()
    wav_any_success = wav.groupby("sentence_id")["success"].any()
    tts_any_failure = tts.groupby("sentence_id")["success"].apply(lambda x: (~x).any())
    wav_any_failure = wav.groupby("sentence_id")["success"].apply(lambda x: (~x).any())

    both_have_success = [s for s in shared_ids if tts_any_success.get(s) and wav_any_success.get(s)]
    both_have_failure = [s for s in shared_ids if tts_any_failure.get(s) and wav_any_failure.get(s)]

    print(f"Shared sentences:            {len(shared_ids)}")
    print(f"Both have >=1 success:       {len(both_have_success)}")
    print(f"Both have >=1 failure:       {len(both_have_failure)}")
    print(f"Overlap between pools:       {len(set(both_have_success) & set(both_have_failure))}")

    # Sample fail pool first (smaller), then success from remaining to avoid duplicates
    n_f = min(args.n_failed,  len(both_have_failure))
    sel_failed = sorted(rng.choice(both_have_failure, size=n_f, replace=False).tolist())

    remaining_success = [s for s in both_have_success if s not in set(sel_failed)]
    n_s = min(args.n_success, len(remaining_success))
    sel_success = sorted(rng.choice(remaining_success, size=n_s, replace=False).tolist())

    all_selected = sorted(sel_success + sel_failed)

    print(f"\nSelected {n_s} both_success + {n_f} both_fail = {len(all_selected)} sentences")
    print(f"  Both-success IDs: {sel_success}")
    print(f"  Both-fail    IDs: {sel_failed}")

    # ------------------------------------------------------------------
    # Copy clips + build manifest
    # ------------------------------------------------------------------
    manifest = []
    sentence_groups = []

    sel_failed_set  = set(sel_failed)
    sel_success_set = set(sel_success)

    for sid in all_selected:
        is_success = sid in sel_success_set
        bucket = "both_success" if is_success else "both_fail"

        tts_row = random_success_run(tts, sid, rng) if is_success else random_failed_run(tts, sid, rng)
        wav_row = random_success_run(wav, sid, rng) if is_success else random_failed_run(wav, sid, rng)

        tts_run_dir = run_folder(args.tts_root,      sid, int(tts_row.run_id))
        wav_run_dir = run_folder(args.waveform_root, sid, int(wav_row.run_id))

        clips = [
            ("GT",       f"gt_{sid:03d}.wav",       os.path.join(tts_run_dir, "ground_truth.wav"), None),
            ("TTS",      f"tts_{sid:03d}.wav",      os.path.join(tts_run_dir, "best_mixed.wav"),   tts_row),
            ("Waveform", f"waveform_{sid:03d}.wav", os.path.join(wav_run_dir, "best_mixed.wav"),   wav_row),
        ]

        gt_text = str(tts_row.ground_truth_text)
        group = {
            "id": sid,
            "type": bucket,
            "gt_text": gt_text,
            "tts_transcription":      str(tts_row.asr_transcription),
            "waveform_transcription": str(wav_row.asr_transcription),
            "tts_pesq":               float(tts_row.pesq),
            "tts_set_overlap":        float(tts_row.set_overlap),
            "waveform_pesq":          float(wav_row.pesq),
            "waveform_set_overlap":   float(wav_row.set_overlap),
        }

        for method, fname, src, row in clips:
            dst = os.path.join(args.output_dir, method, fname)
            ok = copy_clip(src, dst)
            if not ok:
                continue

            rel_path = f"{method}/{fname}"
            group[method.lower()] = rel_path

            entry = {
                "filename": fname,
                "path": rel_path,
                "sentence_id": sid,
                "method": method,
                "bucket": bucket,
                "ground_truth_text": gt_text,
            }
            if row is not None:
                entry.update({
                    "run_id": int(row.run_id),
                    "success": bool(row.success),
                    "pesq": float(row.pesq),
                    "set_overlap": float(row.set_overlap),
                    "asr_transcription": str(row.asr_transcription),
                })
            manifest.append(entry)

            label = f"success={row.success}, PESQ={row.pesq:.3f}" if row is not None else "GT"
            print(f"  [{method:8s}] {fname}  (sentence {sid}, {bucket}, {label})")

        sentence_groups.append(group)

    # ------------------------------------------------------------------
    # Save manifest + JS snippet
    # ------------------------------------------------------------------
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, cls=NumpyEncoder)

    js_path = os.path.join(args.output_dir, "sentence_groups.js")
    with open(js_path, "w") as f:
        f.write("// Auto-generated by select_survey_samples.py\n")
        f.write("// Loaded by index.html via <script src>\n\n")
        f.write("const sentenceGroups = ")
        f.write(json.dumps(sentence_groups, indent=2, cls=NumpyEncoder))
        f.write(";\n")

    print(f"\n[Done] {len(manifest)} clips in {args.output_dir}/")
    print(f"[Done] Manifest → {manifest_path}")
    print(f"[Done] JS array  → {js_path}")


if __name__ == "__main__":
    main()
