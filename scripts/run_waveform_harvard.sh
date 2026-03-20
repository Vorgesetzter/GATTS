#!/bin/bash
# Run adversarial waveform attack on Harvard sentences 1–100
# Run from project root: bash scripts/run_waveform_harvard.sh

# GPU 0: sentences 1–50 (background)
python scripts/adversarial_waveform_harvard.py \
    --sentence_start 16 \
    --sentence_end 50 \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --noise_scale 0.05 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target \
    --gpu 0 &

PID0=$!

# GPU 1: sentences 51–100 (background)
python scripts/adversarial_waveform_harvard.py \
    --sentence_start 70 \
    --sentence_end 100 \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --noise_scale 0.05 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target \
    --gpu 1 &

PID1=$!

wait $PID0 $PID1
