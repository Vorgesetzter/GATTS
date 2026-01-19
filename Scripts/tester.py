import torch

from Models.styletts2 import StyleTTS2
from Models.whisper import Whisper

from Objectives.FitnessObjective import FitnessObjective
from Trainer.EnvironmentLoader import EnvironmentLoader
from Datastructures.dataclass import ModelData, ObjectiveContext
from Datastructures.enum import AttackMode

import os
os.chdir("..")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    active_objectives = ["WER_GT", "PESQ"]
    active_objectives = [obj for obj in FitnessObjective if obj in active_objectives]

    mode = AttackMode.TARGETED

    loader = EnvironmentLoader(device)

    tts = StyleTTS2()
    asr = Whisper()

    text_1 = "I think the NFL is lame and boring"
    text_2 = "The Los Angeles Rams are the worst Team in the world"

    noise = torch.randn(1, 1, 256).to(device)

    token_1 = tts.preprocess_text(text_1)
    token_2 = tts.preprocess_text(text_2)

    audio_1 = tts.inference_on_token(token_1, noise)
    audio_2 = tts.inference_on_token(token_2, noise)

    objectives = loader.initialize_objectives(
        active_objectives=active_objectives,
        model_data=ModelData(tts_model=tts, asr_model=asr),
        text_gt=text_1,
        text_target=text_2,
        mode=mode,
        audio_gt=audio_1,
    )

    asr_1 = asr.inference(audio_1)
    asr_2 = asr.inference(audio_2)

    print(f"ASR 1: {asr_1}")
    print(f"ASR 2: {asr_2}")

if __name__ == "__main__":
    main()