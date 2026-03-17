# GenAdvTTS — Setup & Reproduction Guide

## Installation

> **Recommended:** use a virtual environment (`python -m venv .venv`) or conda before installing.

```bash
pip install -r requirements.txt
sudo apt-get install espeak-ng
```

## Checkpoint

Download the StyleTTS2 LJSpeech checkpoint from [HuggingFace](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main/Models/LJSpeech):

```bash
wget -O checkpoints/STT2.pth https://huggingface.co/yl4579/StyleTTS2-LJSpeech/blob/main/Models/LJSpeech/epoch_2nd_00100.pth
```

## Running the Paper Experiments

Each script splits the 100 Harvard sentences across two GPUs and runs them in parallel.

**Adversarial TTS (main experiment):**
```bash
bash scripts/run_adversarial_tts_harvard.sh
```

**Waveform baseline:**
```bash
bash scripts/run_waveform_harvard.sh
```

Both scripts launch two background processes — GPU 0 handles sentences 1–50, GPU 1 handles 51–100 — and wait for both to finish.

## Single-GPU Setup

To run on a single GPU, replace both invocations with one command covering sentences 1–100 and drop the `--gpu` flag:

```bash
python scripts/adversarial_tts_harvard.py \
    --harvard_sentences_start 1 \
    --harvard_sentences_end 100 \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --iv_scalar 0.5 \
    --size_per_phoneme 1 \
    --num_rms_candidates 1 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target
```

The scripts can also be extended for more than two GPUs by adding further background processes with the appropriate `--harvard_sentences_start`, `--harvard_sentences_end`, and `--gpu` values.

## Outputs

Results are saved to `outputs/<objectives>/<timestamp>/` containing:
- `best_mixed.wav` - Best adversarial audio
- `ground_truth.wav` - Original audio
- `target.wav` - Target audio (if applicable)
- `run_summary.json` - Detailed run report
- `fitness_history.csv` - Fitness values across generations
- Visualization plots (Pareto front, convergence, etc.)
