#!/bin/bash
set -e

SCRIPT="Scripts/adversarial_tts_harvard.py"
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --script) SCRIPT="$2"; shift 2 ;;
        --script=*) SCRIPT="${1#--script=}"; shift ;;
        *) REMAINING_ARGS+=("$1"); shift ;;
    esac
done

echo "Cloning latest code from GitHub..."
git clone https://github.com/Vorgesetzter/StyleTTS2 /app/code

# Make model weights available inside the cloned repo
ln -s /app/Audio /app/code/Audio

cd /app/code
python -u "$SCRIPT" "${REMAINING_ARGS[@]}"
