from ETTS.tester import ETTSInferenceModel
import librosa
import torch
import sys
import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")

_SMACK_DIR = os.path.dirname(os.path.abspath(__file__))

model_syn = ETTSInferenceModel(text_embed_dim=256,
                           emo_embed_dim=768,
                           nmels=80,
                           maxlength=1000,
                           ngst=64,
                           nlst=32,
                           model_dim=256,
                           model_hidden_size=512*2,
                           nlayers=5,
                           nheads=2,
                           vocoder_ckpt_path=os.path.join("checkpoints", 'waveglow_256channels_universal_v5.pt'),
                           etts_checkpoint=os.path.join("checkpoints", 'LJ.ckpt'),
                           sampledir=os.path.join(_SMACK_DIR, 'SampleDir'))


_cached_ref_path = None
_cached_ref_audio = None

def audio_synthesis(l_emo_numpy, reference_audio, reference_text):
    global _cached_ref_path, _cached_ref_audio

    device = model_syn.device

    # Cache reference audio — constant across all evaluations for one sentence
    if reference_audio != _cached_ref_path:
        _cached_ref_audio, sr_global = librosa.load(reference_audio, sr=None)
        assert sr_global == 16000
        _cached_ref_path = reference_audio
    global_audio = _cached_ref_audio

    l_emo = torch.from_numpy(l_emo_numpy).float()
    l_emo = l_emo.to(device)

    with torch.no_grad():
        audio_tensor_syn = model_syn.synthesize_with_sample_lemo(global_audio, l_emo, reference_text, f'synthesis.wav')
        
        audio_numpy = audio_tensor_syn.squeeze(0).cpu().detach().numpy().astype('int16')
        
    return audio_numpy

# For testing purposes
if __name__ == '__main__':
    
    exp_p0_tmp = np.exp(np.random.randn(8, 32) * 10)
    softmax_p0_tmp = exp_p0_tmp / np.sum(exp_p0_tmp, axis=-1, keepdims=True)
    p_0 = softmax_p0_tmp * 0.25
    
    reference_audio = sys.argv[1]
    reference_text = sys.argv[2]
    
    audio_numpy = audio_synthesis(p_0, reference_audio, reference_text)
    