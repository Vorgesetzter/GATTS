import os
import datetime
import platform
import torch
import numpy as np
import soundfile as sf
import pandas as pd

# Local Imports
from Datastructures.dataclass import ConfigData, ModelData, AudioData, BestMixedAudio
from Datastructures.enum import AttackMode
from Trainer.GraphPlotter import GraphPlotter
from helper import adjustInterpolationVector, send_whatsapp_notification, get_pareto_mask

class RunLogger:
    def __init__(self, optimizer, config_data: ConfigData, model_data: ModelData, audio_data: AudioData, fitness_data: list[np.ndarray], gen_count: int, elapsed_time: float, device: str):
        """
        Initializes the logger with specific run results.
        """
        self.optimizer = optimizer
        self.config_data = config_data
        self.model_data = model_data
        self.audio_data = audio_data
        self.device = device

        # Store Run Specific Data
        self.fitness_history = fitness_data
        self.gen_count = gen_count
        self.elapsed_time = elapsed_time

        # Initialize Directory Immediately
        self.folder_path = self._setup_output_directory()

        self.graph_plotter = GraphPlotter(self.folder_path, self.config_data.active_objectives, self.gen_count, self.fitness_history)


    def finalize_run(self):
        """
        Saves all results to the already initialized self.folder_path.
        """

        # 1. Store fitness history
        self._save_fitness_history()

        # 1. Visualization
        self.graph_plotter.generate_all_visualizations()

        # 2. Save Baselines
        self._save_baseline_audio()

        # 3. Process Best Candidate
        best_candidate = self._select_best_candidate(self.optimizer.best_candidates)
        best_mixed_audio = self._run_final_inference(best_candidate)

        # 4. Save Results
        sf.write(os.path.join(self.folder_path, "best_candidate.wav"), best_mixed_audio.audio, samplerate=24000)

        self._save_torch_state(best_mixed_audio, best_candidate)
        self._write_run_summary(best_mixed_audio, best_candidate, self.gen_count, self.elapsed_time)

        # 5. Notify
        if self.config_data.notify:
            send_whatsapp_notification()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _setup_output_directory(self) -> str:
        """Creates the timestamped output folder and returns its path."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        objectives_str = "_".join([obj.name for obj in self.config_data.active_objectives])

        folder_path = os.path.join("outputs/", objectives_str, timestamp)
        os.makedirs(folder_path, exist_ok=True)

        print(f"[Log] Output directory initialized: {folder_path}")
        return folder_path

    def _save_fitness_history(self):
        """
        Saves the complete history of every individual to 'fitness_history.csv'.
        """
        obj_names = [obj.name for obj in self.config_data.active_objectives]

        # Collect all generations into a single list of DataFrames
        raw_list = []

        for gen_idx, gen_matrix in enumerate(self.fitness_history):
            # gen_matrix shape: (pop_size, num_objectives)

            # Create DataFrame for this generation
            df_gen = pd.DataFrame(gen_matrix, columns=obj_names)
            df_gen["Generation"] = gen_idx + 1
            df_gen["Individual_ID"] = df_gen.index  # ID 0 to 99

            raw_list.append(df_gen)

        # Concatenate everything into one massive table
        df_raw = pd.concat(raw_list, ignore_index=True)

        # Reorder columns nicely: [Generation, ID, WER, PESQ, ...]
        cols = ["Generation", "Individual_ID"] + obj_names
        df_raw = df_raw[cols]

        # Save Single Source of Truth
        csv_path = os.path.join(self.folder_path, "fitness_history.csv")
        df_raw.to_csv(csv_path, index=False)

        print("[Log] Full fitness history saved as fitness_history.csv")

    def _save_baseline_audio(self):
        """Saves GT and Target audio if they exist."""
        sf.write(os.path.join(self.folder_path, "ground_truth.wav"), self.audio_data.audio_gt, samplerate=24000)

        if self.audio_data.audio_target is not None:
            sf.write(os.path.join(self.folder_path, "target.wav"), self.audio_data.audio_target, samplerate=24000)

    def _select_best_candidate(self, candidates):
        if not candidates:
            raise ValueError("Candidate list is empty.")

        f = np.array([c.fitness for c in candidates])

        print(f"\n[Log] Candidates on Pareto Front: {len(candidates)}")

        # 1. Threshold filtering
        satisfied_mask = np.ones(len(candidates), dtype=bool)

        if self.config_data.thresholds:
            for i, obj in enumerate(self.config_data.active_objectives):
                if obj in self.config_data.thresholds:
                    limit = self.config_data.thresholds[obj]
                    satisfied_mask &= (f[:, i] <= limit)

        # 2. Logic: If some satisfy thresholds, only pick from those.
        # Otherwise, pick from everyone (fallback).
        if np.any(satisfied_mask):
            eligible_indices = np.where(satisfied_mask)[0]
            eligible_fitness = f[satisfied_mask]
            print(f"[Log] Using {len(eligible_indices)} candidate(s) that meet all thresholds")
        else:
            eligible_indices = np.arange(len(candidates))
            eligible_fitness = f
            print(f"[Log] No candidate met thresholds. Preceding with all candidates.")

        pareto_mask = get_pareto_mask(eligible_fitness)
        final_indices = eligible_indices[pareto_mask]
        final_fitness = eligible_fitness[pareto_mask]

        # --- 3. Knee Point Selection (Balance) ---
        # Normalize to [0,1] range so large metrics (like PESQ) don't dominate small ones (like WER)
        mins = final_fitness.min(axis=0)
        maxs = final_fitness.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0  # Avoid divide by zero

        normalized_fitness = (final_fitness - mins) / ranges

        # Calculate Euclidean distance to the ideal point (0,0,0)
        distances = np.linalg.norm(normalized_fitness, axis=1)
        best_local_idx = np.argmin(distances)

        # Retrieve the original candidate object
        best_global_idx = final_indices[best_local_idx]
        selected = candidates[best_global_idx]

        print(f"[Log] Selected Candidate Fitness: {selected.fitness.tolist()}")
        return selected

    def _run_final_inference(self, best_candidate):

        phoneme_count = int(self.audio_data.input_lengths.item())

        best_vector = torch.from_numpy(best_candidate.solution).to(self.device).float()
        best_vector = best_vector.view(phoneme_count, self.config_data.size_per_phoneme)
        best_vector = adjustInterpolationVector(best_vector, self.config_data.random_matrix, self.config_data.subspace_optimization)

        if self.config_data.mode in [AttackMode.NOISE_UNTARGETED, AttackMode.TARGETED]:
            h_text_mixed = (1.0 - best_vector) * self.audio_data.h_text_gt + best_vector * self.audio_data.h_text_target
        else:
            h_text_mixed = self.audio_data.h_text_gt + self.config_data.iv_scalar * best_vector

        with torch.no_grad():
            audio_best = self.model_data.tts_model.inference_on_embedding(
                self.audio_data.input_lengths,
                self.audio_data.text_mask,
                self.audio_data.h_bert_gt,
                h_text_mixed,
                self.audio_data.style_vector_acoustic,
                self.audio_data.style_vector_prosodic
            )

        if isinstance(self.model_data.asr_model, torch.nn.DataParallel):
            asr_model = self.model_data.asr_model.module
        else:
            asr_model = self.model_data.asr_model

        # 2. Now call transcribe on the unwrapped model
        asr_text = asr_model.transcribe(audio_best)["text"]

        return BestMixedAudio(
            audio=audio_best,
            text=asr_text,
            h_text=h_text_mixed,
            h_bert=self.audio_data.h_bert_gt
        )

    def _save_torch_state(self, best_mixed, candidate):
        """
        Saves all tensors required to reconstruct the adversarial audio
        without re-running the optimization.
        """
        state_dict = {
            # 1. Metadata for reference
            "metadata": {
                "attack_mode": self.config_data.mode.name,
                "text_gt": self.config_data.text_gt,
                "text_target": self.config_data.text_target,
                "asr_transcription": best_mixed.text,
                "fitness_scores": dict(zip([obj.name for obj in self.config_data.active_objectives],
                                           candidate.fitness.tolist()))
            },

            # 2. The Solution (Raw Optimization Result)
            "solution_vector": torch.from_numpy(candidate.solution).float().cpu(),

            # 3. Structural requirements for reconstruction
            "random_matrix": self.config_data.random_matrix.cpu(),
            "size_per_phoneme": self.config_data.size_per_phoneme,
            "iv_scalar": self.config_data.iv_scalar,

            # 4. Model Inputs (AudioData)
            # These are required by the TTS decoder to generate the audio
            "input_lengths": self.audio_data.input_lengths.cpu(),
            "text_mask": self.audio_data.text_mask.cpu(),
            "style_vector_acoustic": self.audio_data.style_vector_acoustic.cpu(),
            "style_vector_prosodic": self.audio_data.style_vector_prosodic.cpu(),

            # 5. The Mixed Embeddings (The final "Adversarial" embeddings)
            "h_text_mixed": best_mixed.h_text.cpu(),
            "h_bert_mixed": best_mixed.h_bert.cpu()
        }

        save_path = os.path.join(self.folder_path, "reconstruction_pack.pt")
        torch.save(state_dict, save_path)
        print("[Log] Torch state saved as reconstruction_pack.pt")

    def _write_run_summary(self, best_mixed, candidate, gen_count, elapsed_time):

        # 1. System Metadata
        os_info = f"{platform.system()} {platform.release()}"
        gpu_info = "CPU Only"
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            gpu_info = f"{torch.cuda.get_device_name(0)} ({vram:.2f} GB VRAM)"

        avg_per_gen = elapsed_time / gen_count if gen_count > 0 else 0

        summary_path = os.path.join(self.folder_path, "run_summary.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write(" ADVERSARIAL TTS OPTIMIZATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("--- [1] INPUT DATA ---\n")
            # Assuming these paths are stored in your config or audio_data
            f.write(f"GT Text:      {self.config_data.text_gt}\n")
            f.write(f"Target Text:  {self.config_data.text_target if self.config_data.text_target else '[NONE]'}\n")

            f.write("\n--- [2] CLI ARGUMENTS & CONFIG ---\n")
            f.write(f"Attack Mode:       {self.config_data.mode.name}\n")
            f.write(f"Objectives:        {', '.join([obj.name for obj in self.config_data.active_objectives])}\n")
            f.write(f"Population Size:   {self.config_data.pop_size}\n")
            f.write(f"Size Per Phoneme:  {self.config_data.size_per_phoneme}\n")
            f.write(f"IV Scalar:         {self.config_data.iv_scalar}\n")
            f.write(f"Subspace Opt:      {self.config_data.subspace_optimization}\n")

            if self.config_data.thresholds:
                t_str = ", ".join([f"{k.name} <= {v}" for k, v in self.config_data.thresholds.items()])
                f.write(f"Early Stopping:    {t_str}\n")
            else:
                f.write(f"Early Stopping:    Off (Ran full duration)\n")

            f.write("\n--- [3] PERFORMANCE & HARDWARE ---\n")
            f.write(f"Hardware:          {gpu_info}\n")
            f.write(f"OS/CPU:            {os_info} | {platform.processor()}\n")
            f.write(f"Gens Completed:    {gen_count}\n")
            f.write(f"Total Time:        {elapsed_time:.2f}s\n")
            f.write(f"Efficiency:        {avg_per_gen:.2f}s per generation\n")

            f.write("\n--- [4] BEST CANDIDATE RESULTS ---\n")
            f.write(f"Selection Metric:  Euclidean Distance to Origin (Knee Point)\n")
            f.write(f"Generation Found:  {getattr(candidate, 'generation', 'Unknown')}\n")
            f.write("-" * 30 + "\n")

            # Detailed Fitness Scores
            for obj, score in zip(self.config_data.active_objectives, candidate.fitness):
                f.write(f"  {obj.name:<15}: {float(score):.8f}\n")

            f.write("-" * 30 + "\n")
            f.write(f"Final Transcription: \"{best_mixed.text}\"\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write(" END OF REPORT\n")
            f.write("=" * 50 + "\n")

