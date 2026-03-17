import torch
from whisper.tokenizer import get_tokenizer
from ..base_objective import BaseObjective
from ...data.dataclass import ObjectiveContext


class WhisperProbGtObjective(BaseObjective):
    """
    Whisper probability ground-truth objective - measures how likely Whisper thinks
    the audio still matches the original ground truth text.

    This objective computes:
    - Cross-entropy loss between predicted logits and ground truth tokens
    - Converted to probability via exp(-loss)
    - Fitness = prob (NOT inverted)

    Values: [0, 1]
    0 = ASR output no longer matches ground truth at all (good - attack succeeded)
    1 = ASR output perfectly matches ground truth (bad - attack failed)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get the actual ASR model (unwrap DataParallel if needed)
        self._real_asr_model = self._get_real_asr_model(self.model_data.asr_model)

        # Prepare tokenized ground truth text for probability computation
        self._gt_tokens_template = self._prepare_whisper_tokens()

    def _get_real_asr_model(self, asr_model):
        """Extract actual model from DataParallel wrapper if needed."""
        if isinstance(asr_model, torch.nn.DataParallel):
            return asr_model.module
        return asr_model

    def _prepare_whisper_tokens(self) -> torch.Tensor:
        """Prepare tokenized ground truth text for WHISPER_PROB_GT computation."""
        tokenizer = get_tokenizer(self._real_asr_model.model.is_multilingual)
        gt_ids = (
            list(tokenizer.sot_sequence) +
            tokenizer.encode(self.text_gt) +
            [tokenizer.eot]
        )
        return torch.tensor([gt_ids]).to(self.device)

    @property
    def supports_batching(self) -> bool:
        return True

    def _calculate_logic(self, context: ObjectiveContext) -> list[float]:
        """
        Computes the whisper ground truth probability fitness values from the mel spectrogram.

        Args:
            context: ObjectiveContext containing mel_batch

        Returns:
            List of fitness values (0 = best, 1 = worst)
        """
        if context.mel_batch is None:
            raise ValueError(
                "WhisperProbGtObjective requires mel_batch in ObjectiveContext. "
                "Ensure the main loop provides mel spectrogram before calling this objective."
            )

        mel_batch = context.mel_batch
        batch_size = mel_batch.shape[0]

        # Expand ground truth tokens to batch size
        gt_tokens_batch = self._gt_tokens_template.expand(batch_size, -1)

        with torch.no_grad():
            logits = self._real_asr_model.model(mel_batch, gt_tokens_batch)

        # Shift for causal LM alignment: predict token[i+1] from token[i]
        logits_shifted = logits[:, :-1, :]
        targets_shifted = gt_tokens_batch[:, 1:]

        # Cross-entropy loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        raw_losses = loss_fct(
            logits_shifted.reshape(-1, logits_shifted.size(-1)),
            targets_shifted.reshape(-1)
        )

        # Average loss over sentence length
        sample_losses = raw_losses.reshape(batch_size, -1).mean(dim=1)

        # Fitness = probability of matching GT (minimize to succeed)
        probs = torch.exp(-sample_losses)

        return probs.detach().cpu().tolist()
