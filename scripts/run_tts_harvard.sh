#!/bin/bash
# Run from project root: bash scripts/run_adversarial_tts_harvard.sh

# GPU 0: sentences 1–50 (background)
python scripts/adversarial_tts_harvard.py \
    --harvard_sentences_start 13 \
    --harvard_sentences_end 50 \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --iv_scalar 0.5 \
    --size_per_phoneme 1 \
    --num_rms_candidates 1 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target \
    --gpu 0 &

PID0=$!

# GPU 1: sentences 51–100 (background)
python scripts/adversarial_tts_harvard.py \
    --harvard_sentences_start 67 \
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
    --seed_target \
    --gpu 1 &

PID1=$!

wait $PID0 $PID1
