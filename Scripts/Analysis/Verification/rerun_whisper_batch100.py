"""
Re-run Whisper at batch_size=100 on all existing runs and update run_summary.json.

Old runs stored the ASR transcription from run_final_inference at batch_size=1.
The SET_OVERLAP score in fitness_scores was computed during optimization at batch_size=100.
These disagree due to Whisper's cuBLAS GEMM kernel selection being batch-size dependent.

This script re-runs Whisper at batch_size=100 on every best_mixed.wav and updates:
  - text_data.asr_transcription
  - success_metrics.fitness_scores.SET_OVERLAP
  - success_metrics.success

Run on the same GPU type used for optimization for best reproducibility.

Usage (local / Vertex AI Workbench):
  python Scripts/Analysis/rerun_whisper_batch100.py

Usage (Vertex AI custom job, data on GCS):
  python Scripts/Analysis/rerun_whisper_batch100.py \
      --tts_root      gs://your-bucket/outputs/results/TTS \
      --waveform_root gs://your-bucket/outputs/results/Waveform \
      --gcs

When --gcs is set the script downloads each run folder to /tmp, processes it,
then uploads the updated run_summary.json back to GCS.
"""

import argparse
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torchaudio
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from Models.whisper import Whisper

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOPWORDS  = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

SET_OVERLAP_THRESHOLD = 0.5
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# NLP helpers
# ---------------------------------------------------------------------------

def content_lemmas(text: str) -> set:
    clean = re.sub(r"[^\w\s]", "", text.lower())
    words = set(clean.split()) - STOPWORDS
    result = set()
    for w in words:
        for pos in ("a", "v", "n", "r"):
            lemma = LEMMATIZER.lemmatize(w, pos=pos)
            if lemma != w:
                result.add(lemma)
                break
        else:
            result.add(w)
    return result


def set_overlap(gt: str, asr: str) -> float:
    gt_l = content_lemmas(gt)
    return len(gt_l & content_lemmas(asr)) / len(gt_l) if gt_l else 1.0


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform  # (1, samples)


# ---------------------------------------------------------------------------
# GCS helpers (uses google-cloud-storage Python library, already in image)
# ---------------------------------------------------------------------------

def _gcs_parse(gcs_path: str):
    """Split gs://bucket/blob into (bucket, blob)."""
    assert gcs_path.startswith("gs://"), f"Not a GCS path: {gcs_path}"
    parts = gcs_path[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def gcs_download(gcs_path: str, local_path: str):
    from google.cloud import storage
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    bucket_name, blob_name = _gcs_parse(gcs_path)
    storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(local_path)


def gcs_upload(local_path: str, gcs_path: str):
    from google.cloud import storage
    bucket_name, blob_name = _gcs_parse(gcs_path)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(local_path)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_run(run_path: str, wav_path: str, summary_path: str,
                asr_model, device: str) -> bool:
    """Update a single run. Returns True if the transcription changed."""
    with open(summary_path) as f:
        summary = json.load(f)

    gt_text = summary["text_data"]["ground_truth_text"]
    old_asr = summary["text_data"].get("asr_transcription", "")

    audio = load_audio(wav_path).to(device).expand(BATCH_SIZE, -1)
    texts, _ = asr_model.inference(audio)
    new_asr   = texts[0]
    new_score = set_overlap(gt_text, new_asr)

    changed = new_asr != old_asr
    old_score = summary["success_metrics"]["fitness_scores"].get("SET_OVERLAP")

    summary["text_data"]["asr_transcription"] = new_asr
    summary["success_metrics"]["fitness_scores"]["SET_OVERLAP"] = new_score

    # Recompute success using all stored thresholds where available
    thresholds = summary.get("algorithm_parameters", {}).get("thresholds", {})
    scores     = summary["success_metrics"]["fitness_scores"]
    if thresholds:
        new_success = all(scores.get(k, float("inf")) <= v for k, v in thresholds.items())
    else:
        new_success = new_score < SET_OVERLAP_THRESHOLD
    summary["success_metrics"]["success"] = new_success

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if changed:
        old_s = f"{old_score:.3f}" if old_score is not None else "n/a"
        print(f"  [CHANGED] {os.path.relpath(run_path)}")
        print(f"    old asr:   {old_asr}")
        print(f"    new asr:   {new_asr}")
        print(f"    score:     {old_s} -> {new_score:.3f}  success={new_success}")
    else:
        print(f"  [same]    {os.path.relpath(run_path)}")

    return changed


def iter_runs_local(root: str):
    """Yield (run_path, wav_path, summary_path) for each run under root."""
    if not os.path.isdir(root):
        print(f"[SKIP] {root} not found")
        return
    for sent in sorted(os.listdir(root)):
        if not sent.startswith("sentence_"):
            continue
        sent_path = os.path.join(root, sent)
        for run in sorted(os.listdir(sent_path)):
            run_path     = os.path.join(sent_path, run)
            wav_path     = os.path.join(run_path, "best_mixed.wav")
            summary_path = os.path.join(run_path, "run_summary.json")
            if os.path.exists(wav_path) and os.path.exists(summary_path):
                yield run_path, wav_path, summary_path


def iter_runs_gcs(gcs_root: str):
    """List GCS run folders and yield (gcs_run, local_wav, local_summary, gcs_summary)."""
    from google.cloud import storage
    bucket_name, prefix = _gcs_parse(gcs_root)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all run_summary.json blobs under the prefix
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    summaries = [b.name for b in blobs if b.name.endswith("/run_summary.json")]

    for blob_name in sorted(summaries):
        gcs_summary = f"gs://{bucket_name}/{blob_name}"
        gcs_run     = f"gs://{bucket_name}/{blob_name.rsplit('/', 1)[0]}"
        gcs_wav     = gcs_run + "/best_mixed.wav"

        # Check wav exists
        wav_blob_name = blob_name.rsplit("/", 1)[0] + "/best_mixed.wav"
        if not bucket.blob(wav_blob_name).exists():
            continue

        tmp_dir   = tempfile.mkdtemp()
        local_wav = os.path.join(tmp_dir, "best_mixed.wav")
        local_sum = os.path.join(tmp_dir, "run_summary.json")
        try:
            gcs_download(gcs_wav,     local_wav)
            gcs_download(gcs_summary, local_sum)
        except Exception as e:
            print(f"[ERROR downloading {gcs_run}]: {e}")
            continue
        yield gcs_run, local_wav, local_sum, gcs_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts_root",      default="outputs/results/TTS")
    parser.add_argument("--waveform_root", default="outputs/results/Waveform")
    parser.add_argument("--gcs",           action="store_true",
                        help="Roots are GCS paths (gs://...); use gsutil for I/O")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  batch_size: {BATCH_SIZE}")
    asr_model = Whisper(device=device)

    total = changed = errors = 0

    for root in [args.tts_root, args.waveform_root]:
        print(f"\n=== Processing {root} ===")

        if args.gcs:
            for gcs_run, wav_path, summary_path, gcs_summary in iter_runs_gcs(root):
                try:
                    was_changed = process_run(gcs_run, wav_path, summary_path, asr_model, device)
                    if was_changed:
                        gcs_upload(summary_path, gcs_summary)
                        changed += 1
                    total += 1
                except Exception as e:
                    print(f"  [ERROR] {gcs_run}: {e}")
                    errors += 1
        else:
            for run_path, wav_path, summary_path in iter_runs_local(root):
                try:
                    if process_run(run_path, wav_path, summary_path, asr_model, device):
                        changed += 1
                    total += 1
                except Exception as e:
                    print(f"  [ERROR] {run_path}: {e}")
                    errors += 1

    print(f"\n{'='*60}")
    print(f"Total runs processed : {total}")
    print(f"Transcription changed: {changed}")
    print(f"Errors               : {errors}")


if __name__ == "__main__":
    main()
