from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
# from lippy.utils.speaker import Speaker
# from lippy.utils.bark_test import *
from nltk import sent_tokenize
import numpy as np
# from scipy.io import wavfile
import os
from IPython.display import Audio
from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE

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
    # wavfile.write(path, samp, audio_output)

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


base_dir = "/home/ubuntu/Tehas/"
pathAudio = base_dir + "lippy/data/audio"
samp = SAMPLE_RATE
pathLib = "/home/ubuntu/Tehas/lippy/data/books/"
bookName = "Vagabonding - Rolf Potts"
pathBook = pathLib + bookName + ".epub"
GEN_TEMP = 0.6
book = epub.read_epub(pathBook)
items = list(book.get_items_of_type(ITEM_DOCUMENT))
collection = []
# print([item.get_name() for item in items])
for i, item in enumerate(items[7:]):
    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
    text = [para.get_text() for para in soup.find_all('p')]
    text = "" ' '.join(text)
    if len(text) ==0:
        continue
    fn =  f"Vagabond_{i}"
    bark_speak(text, fn)
    collection.append(text)
    print(fn)
    print(text)
# soup = BeautifulSoup(items[8].get_body_content(), 'html.parser')
# text = [para.get_text() for para in soup.find_all('p')]
# text = "" ' '.join(text)
# for t in range (5,9):
#     t = t * .1
#     for w in range(5,9):
#         w = w * .1
#         bark_speak(text[:500], f"Vagabond_preface_t{t}_w{w}",temp=t, wave_temp=w)


#     if i > 5: break
# voice = Speaker("/home/theatasigma/lippy/data/audio")
# print(sample_text)
# voice.say(sample_text)