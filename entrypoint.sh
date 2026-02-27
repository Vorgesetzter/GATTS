#!/bin/bash
set -e

echo "Cloning latest code from GitHub..."
git clone https://github.com/Vorgesetzter/StyleTTS2 /app/code

# Make model weights available inside the cloned repo
ln -s /app/Audio /app/code/Audio

cd /app/code
python -u Scripts/adversarial_tts_harvard.py "$@"
