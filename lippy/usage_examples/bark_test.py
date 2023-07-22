from nltk import sent_tokenize,download
import numpy as np
from bark import generate_audio, SAMPLE_RATE
import os
from IPython.display import Audio
# download('punkt')
from lippy.utils.listener import Listener

def save_audio(
    audio_output,
    base_dir: str = "/home/ubuntu/Tehas/",
    filename: str = "output.wav",
    overwrite: bool = False
):
    """
    Saves the audio as a WAV file.

    Parameters:
    audio_output (torch.Tensor): The audio to be saved.
    filename (str): The filename for the saved audio.
    overwrite (bool, optional): Whether to overwrite an existing file
                                with the same name.
                                Defaults to False.
    """
    path = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not overwrite:
        print(f"File {path} already exists.")
        return

    print(f'Saving {filename} to {base_dir}')
    op = Audio(audio_output, rate=SAMPLE_RATE)
    print(op)
    with open(path, 'wb') as f:
        f.write(op.data)

def bark_speak(text,speaker="v2/en_speaker_3", output_fn="bark_output", temp=0.8, wave_temp=0.8):
    print(f"[BARK] speak: {text[:100]}...")
    sentences = sent_tokenize(text)
    print(f"[BARK] num tokens: {len(sentences)}")
    # speaker = "/home/ubuntu/Tehas/lippy/voices/Dalinar-4_t7_w7_6.npz"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))
    pieces = []
    for i, sent in enumerate(sentences):
        audio_array = generate_audio(sent, history_prompt=speaker, text_temp=temp, waveform_temp=wave_temp)
        pieces += [audio_array, silence.copy()]
    print(f"[BARK] wav: {np.concatenate(pieces)}")
    save_audio(np.concatenate(pieces), pathBase, f"{output_fn}.wav",True)

pathBase = "/home/ubuntu/Tehas/lippy/data/audio/"
pathOP = "bark_test_op"
pathAudio = pathBase + pathOP
samp = SAMPLE_RATE
sample_text = 'So when one student complained to my teacher Ajahn Chah that in his very busy life he did not have time to meditate, Ajahn Chah laughed and said, "Do you have time to breathe? If you are determined, you must simply pay attention. This is our practice, wherever you are, whatever is happening: to breathe, to be fully present, to see what is true.'
bark_speak(sample_text, "/home/ubuntu/Tehas/lippy/voices/StormFather_seed-1.npz", pathOP)
listener = Listener()
promptTranscript = listener.transcribe(pathAudio + ".wav")
print(f"Ground Truth: {sample_text}")
print(f"Transcription: {promptTranscript['text']}")
# for root, dirs, files in os.walk("/home/ubuntu/Tehas/lippy/voices/"):
#     for file in files:
#         # print(root+file)
#         bark_speak(sample_text, speaker=root+file, output_fn=f"wit_mem_{file}")