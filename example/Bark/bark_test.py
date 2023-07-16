import nltk
import numpy as np
from scipy.io import wavfile
# from bark.generation import generate_text_semantic, preload_models
# bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
import os
from IPython.display import Audio

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

def bark_speak(text, output_fn="bark_output", temp=0.8, wave_temp=0.8):
    print(f"[BARK] speak: {text[:100]}...")
    sentences = sent_tokenize(text)
    print(f"[BARK] num tokens: {len(sentences)}")
    speaker = "v2/en_speaker_3"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))
    pieces = []
    for i, sent in enumerate(sentences):
        audio_array = generate_audio(sent, history_prompt=speaker, text_temp=temp, waveform_temp=wave_temp)
        pieces += [audio_array, silence.copy()]
    print(f"[BARK] wav: {np.concatenate(pieces)}")
    save_audio(np.concatenate(pieces), pathAudio, f"{output_fn}.wav",True)

base_dir = "/home/ubuntu/Tehas/lippy/data/audio"
samp = SAMPLE_RATE
sample_text = "[Lovingly, passionately] Memory is often our only connection to who we used to be. Memories are fossils, the bones left by dead versions of ourselves. More potently, our minds are a hungry audience, craving only the peaks and valleys of experience. The bland erodes, leaving behind distinctive bits to be remembered again and again. Painful or passionate, surreal or sublime, we cherish those little rocks of peak experience, polishing them with the ever-smoothing touch of recycled proxy living. In doing, like pagans praying to a sculpted mud figure, we make our memories the gods which judge our current lives."
bark_speak(sample_text, "wit_mem")