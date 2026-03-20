import sys
import io
import os
from .NISQA_model import nisqaModel

_nisqa_instance = None
_nisqa_model_path = None

def NISQA_score(audio_file, pretrained_model='nisqa_tts.tar', ms_channel=None, tr_bs_val=1, tr_num_workers=0, output_dir=None):
    global _nisqa_instance, _nisqa_model_path

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_model)

    # divert stdout to null stream
    null_stream = io.StringIO()
    sys.stdout = null_stream

    if _nisqa_instance is None or _nisqa_model_path != model_path:
        _nisqa_instance = nisqaModel({
            'mode': 'predict_file',
            'pretrained_model': model_path,
            'deg': audio_file,
            'ms_channel': ms_channel,
            'tr_bs_val': tr_bs_val,
            'tr_num_workers': tr_num_workers,
            'output_dir': output_dir
        })
        _nisqa_model_path = model_path
    else:
        _nisqa_instance.args['deg'] = audio_file
        _nisqa_instance._loadDatasets()

    nisqa_res = _nisqa_instance.predict()

    # divert stream back
    sys.stdout = sys.__stdout__

    return nisqa_res.iloc[0]['mos_pred']

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        print(NISQA_score(audio_file))
    else:
        print("Usage: python predict.py <audio_file>")