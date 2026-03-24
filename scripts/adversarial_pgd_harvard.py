"""
Adversarial PGD — Harvard Sentences Experiment

Loads pre-generated reference audios and runs the PGD untargeted attack
(from whisper_attack) on each Harvard sentence.

Generate reference audios first (styletts2 env):
    python scripts/generate_harvard_audios.py --start 1 --end 100

Then run this script (whisper_attack env) from project root:
    python scripts/adversarial_pgd_harvard.py --start 1 --end 100
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import shutil
import argparse
import subprocess
import soundfile as sf
import torch

from src.data.harvard_sentences import HARVARD_SENTENCES
from src.models._whisper import Whisper
from src.trainer.result_writer import save_attack_result


NB_ITER = 200
SEED = 235
SNR = 35

AUDIO_DIR = 'outputs'
WHISPER_ATTACK_DIR = 'scripts/PGD'


def create_csv(sentence_ids, output_dir):
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'harvard.csv')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'duration', 'wav', 'wrd'])
        for sid in sentence_ids:
            audio_path = os.path.abspath(os.path.join(AUDIO_DIR, f'harvard_sentence_{sid:03d}', 'harvard_audio.wav'))
            if not os.path.exists(audio_path):
                print(f"[{sid:3d}] Audio not found, skipping: {audio_path}")
                continue
            duration = sf.info(audio_path).duration
            text = HARVARD_SENTENCES[sid - 1].upper()
            writer.writerow([f'sentence_{sid:03d}', f'{duration:.3f}', audio_path, text])

    return csv_path


def run_pgd_attack(output_dir, gpu=None):
    abs_output_dir = os.path.abspath(output_dir)
    device = f'cuda:{gpu}' if (gpu is not None and torch.cuda.is_available()) else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    cmd = [
        sys.executable, '-W', 'ignore', 'run_attack.py',
        'attack_configs/whisper/pgd.yaml',
        f'--root={abs_output_dir}',
        f'--data_folder={abs_output_dir}',
        '--data_csv_name=harvard',
        f'--nb_iter={NB_ITER}',
        '--load_audio=False',
        f'--seed={SEED}',
        '--attack_name=pgd_harvard',
        f'--snr={SNR}',
        '--skip_prep=True',
        f'--device={device}',
    ]
    env = os.environ.copy()
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print(f"[PGD] Running attack on device={device} | nb_iter={NB_ITER} | snr={SNR}")
    result = subprocess.run(cmd, cwd=WHISPER_ATTACK_DIR, env=env)
    print(f"[PGD] subprocess exited with code {result.returncode}")


def organize_outputs(sentence_ids, whisper_model, elapsed_time_seconds=None, n_sentences=1):
    save_path = os.path.join(AUDIO_DIR, 'pgd_save')

    for sid in sentence_ids:
        adv_src = os.path.join(save_path, f'sentence_{sid:03d}_adv.wav')
        if not os.path.exists(adv_src):
            print(f"[{sid:3d}] Adversarial audio not found, skipping")
            continue

        sentence_dir = os.path.join(AUDIO_DIR, f'harvard_sentence_{sid:03d}')
        adv_dst = os.path.join(sentence_dir, 'pgd.wav')
        shutil.copy(adv_src, adv_dst)

        audio, sr = sf.read(adv_dst)
        texts, _ = whisper_model.inference(torch.from_numpy(audio).float())
        transcription = texts[0]

        sentence_elapsed = elapsed_time_seconds / n_sentences if elapsed_time_seconds else 0.0
        save_attack_result(
            sentence_id=sid,
            method='pgd',
            audio=audio,
            transcription=transcription,
            gt_text=HARVARD_SENTENCES[sid - 1],
            elapsed=sentence_elapsed,
            params={
                'num_generations': NB_ITER,
                'pop_size': 1,
                'snr': SNR,
                'seed': SEED,
            },
        )

    shutil.rmtree(save_path, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='PGD Untargeted Attack — Harvard Sentences')
    parser.add_argument('--start', type=int, default=1, help='First sentence index (1-based)')
    parser.add_argument('--end', type=int, default=100, help='Last sentence index (1-based, inclusive)')
    parser.add_argument('--nb_iter', type=int, default=NB_ITER, help='Number of PGD iterations')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use')
    args = parser.parse_args()

    global NB_ITER
    NB_ITER = args.nb_iter

    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")

    sentence_ids = list(range(args.start, args.end + 1))

    print(f"Sentences: {args.start} → {args.end}")
    print(f"SNR: {SNR} | Iterations: {NB_ITER} | Seed: {SEED}")
    print('=' * 60)

    whisper_model = Whisper()

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    csv_path = create_csv(sentence_ids, output_dir)
    print(f"CSV created: {csv_path}\n")

    import time
    t0 = time.time()
    run_pgd_attack(output_dir)
    elapsed = time.time() - t0

    organize_outputs(sentence_ids, whisper_model, elapsed_time_seconds=elapsed, n_sentences=len(sentence_ids))
    print('\n[Done]')


if __name__ == '__main__':
    main()
