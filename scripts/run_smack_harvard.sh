#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

# Split ranges
START0=1
END0=50

START1=51
END1=100

# ---------- Generate all audios first ----------
python scripts/generate_harvard_audios.py --start $START0 --end $END1

# ---------- GPU 0 ----------
(
conda run -n smack python scripts/adversarial_smack_harvard.py \
    --start $START0 --end $END0 \
    --gpu 0
) &

PID0=$!

# ---------- GPU 1 ----------
(
conda run -n smack python scripts/adversarial_smack_harvard.py \
    --start $START1 --end $END1 \
    --gpu 1
) &

PID1=$!

wait $PID0 $PID1