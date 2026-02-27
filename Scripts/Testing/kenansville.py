import argparse
import os
import sys
import numpy as np
import scipy.io.wavfile as wav
import scipy.fftpack as fft
import librosa
import datetime
import difflib
import torch
import speech_recognition as sr
import soundfile as sf

# --- GLOBAL MODEL CACHE ---
# We store loaded models here so we don't reload them every iteration
LOADED_MODELS = {}


def load_whisper(model_size="tiny.en"):
    """Lazy loader for Whisper to avoid consuming VRAM if not used"""
    if "whisper" not in LOADED_MODELS:
        print(f"[*] Loading Whisper ({model_size})...")
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
        model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}").to(device)
        model.eval()
        LOADED_MODELS["whisper"] = (model, processor, device)
    return LOADED_MODELS["whisper"]


def load_deepspeech(model_path, scorer_path):
    """Lazy loader for DeepSpeech"""
    if "deepspeech" not in LOADED_MODELS:
        try:
            from deepspeech import Model
        except ImportError:
            print("[!] Error: 'deepspeech' library not installed. Run 'pip install deepspeech'")
            sys.exit(1)

        if not model_path or not os.path.exists(model_path):
            print(f"[!] Error: DeepSpeech model file not found at: {model_path}")
            print("    Download it from: https://github.com/mozilla/DeepSpeech/releases")
            sys.exit(1)

        print(f"[*] Loading DeepSpeech model from {model_path}...")
        ds_model = Model(model_path)
        if scorer_path and os.path.exists(scorer_path):
            ds_model.enableExternalScorer(scorer_path)

        LOADED_MODELS["deepspeech"] = ds_model
    return LOADED_MODELS["deepspeech"]


def clean_text(text):
    """Standardizes text for comparison (remove punctuation, lower case)"""
    if text is None: return ""
    return text.lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '').strip()


def transcribe(audio_path, model_type, args=None):
    """
    Universal transcription router.
    """
    # 1. WHISPER
    if model_type == 'whisper':
        model, processor, device = load_whisper()
        # Librosa loads as float32 [-1,1], sr=16000 required for Whisper
        audio_array, _ = librosa.load(audio_path, sr=16000)

        input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            generated_ids = model.generate(input_features)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return clean_text(transcription)

    # 2. DEEPSPEECH
    elif model_type == 'deepspeech':
        ds_model = load_deepspeech(args.ds_model, args.ds_scorer)

        # DeepSpeech expects 16kHz int16 audio
        audio_int16, fs = sf.read(audio_path, dtype='int16')
        if fs != 16000:
            # Resample strictly to 16000
            audio_float, _ = librosa.load(audio_path, sr=16000)
            audio_int16 = (audio_float * 32767).astype(np.int16)

        return clean_text(ds_model.stt(audio_int16))

    # 3. CLOUD APIs (Google, Azure) - Uses SpeechRecognition lib
    else:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)

        try:
            if model_type == 'google':
                return clean_text(r.recognize_google(audio_data))

            elif model_type == 'azure':
                if not args.azure_key or not args.azure_region:
                    print("[!] Error: Azure requires --azure_key and --azure_region")
                    sys.exit(1)
                return clean_text(r.recognize_azure(audio_data, key=args.azure_key, location=args.azure_region)[0])

        except sr.UnknownValueError:
            return ""  # Audio unintelligible (Success)
        except sr.RequestError as e:
            print(f"[!] API Error ({model_type}): {e}")
            return "api_error"


def fft_compression(path, audio_image, factor, fs):
    # FFT Attack (Spectral Gating)
    fft_image = fft.fft(audio_image.ravel())

    # Zero out values below threshold (Spectral Gating)
    mask = np.abs(fft_image) < factor
    fft_image[mask] = 0

    ifft_audio = fft.ifft(fft_image).real

    # Temp path
    new_audio_path = path[:-4] + '_temp.wav'
    return new_audio_path, np.asarray(ifft_audio, dtype=np.int16)


def bst_atk_factor(min_atk, max_atk, val_atk, og_label, atk_label):
    """
    Binary search optimization.
    Returns: new_min, new_max, new_val, is_converged
    """
    # Calculate similarity (0.0 to 1.0)
    # If text is < 80% similar, we consider the attack a success.
    similarity = difflib.SequenceMatcher(None, og_label, atk_label).ratio()
    success = (similarity < 0.8)

    init_val = val_atk

    if success:
        # Attack worked too well? Try to be quieter (reduce factor)
        max_atk = val_atk
        val_atk = (min_atk + max_atk) / 2
    else:
        # Attack failed? Hit harder (increase factor)
        min_atk = val_atk
        val_atk = (min_atk + max_atk) / 2

    return int(min_atk), int(max_atk), int(val_atk), (init_val == val_atk)


def run_attack(args):
    print(f"[*] Starting attack on model: {args.model}")

    # Create output dir
    if os.path.dirname(args.outputfile):
        os.makedirs(os.path.dirname(args.outputfile), exist_ok=True)

    # 1. Get Ground Truth
    og_label = transcribe(args.inputfile, args.model, args)
    print(f"[+] Original Label: '{og_label}'")
    if not og_label:
        print("[!] Error: Model returned empty transcription for original file. Aborting.")
        return

    # 2. Load Audio
    fs, data = wav.read(args.inputfile)
    if len(data.shape) > 1: data = data[:, 0]  # Mono

    # 3. Setup Binary Search
    min_f = 0
    # Use max FFT energy as upper bound
    max_f = np.max(np.abs(fft.fft(data)))
    curr_f = max_f / 2

    best_adversarial_audio = np.copy(data)

    # 4. Attack Loop
    for i in range(15):  # Max 15 iterations
        # Apply FFT Gating
        temp_path, adv_frames = fft_compression(args.inputfile, data, curr_f, fs)

        # Save temp file for transcription
        wav.write(temp_path, fs, adv_frames.astype(np.int16))

        # Transcribe
        adv_label = transcribe(temp_path, args.model, args)

        # Calculate distortion (L1 norm normalized)
        distortion = np.mean(np.abs(data - adv_frames)) / np.max(np.abs(data))

        print(f"Iter {i + 1:02d} | Factor: {curr_f:.0f} | Dist: {distortion:.4f}")
        print(f"        -> '{adv_label}'")

        # Check for improvement (if attack worked, save this as candidate)
        similarity = difflib.SequenceMatcher(None, og_label, adv_label).ratio()
        if similarity < 0.8:
            best_adversarial_audio = adv_frames

        # Update Binary Search
        min_f, max_f, curr_f, done = bst_atk_factor(min_f, max_f, curr_f, og_label, adv_label)

        if os.path.exists(temp_path): os.remove(temp_path)
        if done: break

    # 5. Save Final Result
    print(f"[+] Saving result to: {args.outputfile}")
    wav.write(args.outputfile, fs, best_adversarial_audio.astype(np.int16))


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join("outputs/kenansville", f"{timestamp}.wav")

    parser = argparse.ArgumentParser(description="Universal Kenansville Audio Attack")

    # Basic Arguments
    parser.add_argument("-i", "--inputfile", default="synthesized_audio.wav", help="Path to input .wav file")
    parser.add_argument("-o", "--outputfile", default=output_path, help="Path to save output")
    parser.add_argument("-m", "--model", choices=['google', 'whisper', 'deepspeech', 'azure'], default='whisper', help="Target model to attack")

    # DeepSpeech Specifics
    parser.add_argument("--ds_model", help="Path to deepspeech .pbmm model file")
    parser.add_argument("--ds_scorer", help="Path to deepspeech .scorer file")

    # Azure Specifics
    parser.add_argument("--azure_key", help="Microsoft Azure Speech API Key")
    parser.add_argument("--azure_region", help="Azure Region (e.g. eastus)")

    args = parser.parse_args()

    run_attack(args)