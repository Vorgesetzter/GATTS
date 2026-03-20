import sys
import torch
import librosa


from src.models import load_whisper_model

_whisper_model = None

def whisper_ASR(audio_file):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = load_whisper_model("base", device="cuda")
    model = _whisper_model

    # Load audio at native sample rate
    audio, sr = librosa.load(audio_file, sr=None)
    audio_tensor = torch.from_numpy(audio).float()

    # Call inference method (handles resampling internally)
    clean_texts, _ = model.inference(audio_tensor)

    text = clean_texts[0].upper() if clean_texts else ""

    # Normalize: keep only alphanumeric and spaces
    text = "".join(c for c in text if c.isalnum() or c.isspace())

    return text if text else "NA"


# For testing purposes
if __name__ == "__main__":
    audio_file = sys.argv[1]
    result = whisper_ASR(audio_file)
    print(f'Whisper ASR Result: {result}')
