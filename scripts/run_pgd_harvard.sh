#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

# Split attack ranges
START0=1
END0=1

START1=2
END1=2

# ---------- Generate all audios first ----------
python scripts/generate_harvard_audios.py --start $START0 --end $END1

# ---------- CPU (no CUDA) ----------
CUDA_VISIBLE_DEVICES="" conda run -n pgd python -W ignore scripts/adversarial_pgd_harvard.py \
    --start $START0 --end $END1