import os

import whisper


here_dir = os.path.dirname(os.path.abspath(lippy.__file__))
data_dir = os.path.join(here, os.pardir, os.pardir, os.pardir, 'data')
wav_fp = os.path.join(data_dir, 'audio', 'output.wav')

model = whisper.load_model('base')
result = model.transcribe(wav_fp)
print(result['text'])