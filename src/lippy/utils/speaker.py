import os
import torch
from IPython.display import Audio
from nltk import sent_tokenize, download
import numpy as np
from bark import generate_audio, SAMPLE_RATE, preload_models
from os import listdir


class Speaker:
    def __init__(self, backend="bark", speaker_id = "Dalinar-1_t8_w8_7", op_dir: str = "/home/ubuntu/Tehas/lippy/data/audio"):
        """
        Initializes the Speaker class.

        Parameters:
        base_dir (str, optional): The base directory where the audio files will
                                  be saved.
                                  Defaults to "audio_files".
        """
        self.backend= backend
        self.params = {
            "silero":{
                "lang": 'en',
                "model_id": "v3_en",
                "sample_rate": 48000,
                "speaker": speaker_id,
                "device": torch.device('cpu')
            }, "bark": {
                "speaker": speaker_id + ".npz",
                "sample_rate": SAMPLE_RATE,
                "custom_voice_dir": "/home/ubuntu/Tehas/lippy/voices/",
                "voice_dir": "/home/ubuntu/Tehas/lippy/voices/" if speaker_id + ".npz" in listdir("/home/ubuntu/Tehas/lippy/voices/") else "/home/ubuntu/.local/lib/python3.8/site-packages/bark/assets/prompts/v2/"
            }
        }
        self.op_dir = op_dir
        self.load_model()

    def load_model(self):
        """
        Loads the pre-trained TTS model.
        """
        params = self.params[self.backend]
        if self.backend == "silero":
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=params["lang"],
                speaker=params["speaker"]
            )
            self.model.to(self.device)
        elif self.backend == "bark":
            download('punkt')
            preload_models()
        
    def split_into_sentences(self, input_string):
        # Split the input string into individual sentences
        sentences = input_string.split('. ')

        # Initialize variables
        result = []
        current_sentence = ''

        # Iterate over each sentence
        for sentence in sentences:
            # Append the current sentence and the next sentence
            temp = current_sentence + sentence + '. '

            # Check if the combined sentence is within the limit
            if len(temp) <= 900:
                current_sentence = temp
            else:
                # Split the sentence at word boundaries
                words = sentence.split()
                temp = current_sentence

                for word in words:
                    # Check if adding the word exceeds the limit
                    if len(temp) + len(word) <= 900:
                        temp += word + ' '
                    else:
                        # Add the current sentence to the result
                        result.append(temp)
                        temp = word + ' '

                # Add the remaining sentence to the result
                if temp.strip():
                    result.append(temp)

                current_sentence = ''

        # Add the last sentence to the result
        if current_sentence.strip():
            result.append(current_sentence)

        return result

    def say(self, input:str, output_fn="output", temp=0.8, wave_temp=0.8):
        params = self.params[self.backend]
        if self.backend == "silero":
            strings = self.split_into_sentences(input)
            outputs = []
            for i, sent in enumerate(strings):
                outputTensor = self.model.apply_tts(
                    text=sent,
                    speaker=params["speaker"],
                    sample_rate=params["sample_rate"]
                )
                outputs.append(outputTensor)
            outputs = torch.cat(outputs)

        elif self.backend == "bark":
            print(f"[BARK] speak: {input[:100]}...")
            sentences = sent_tokenize(input)
            print(f"[BARK] num tokens: {len(sentences)}")
            silence = np.zeros(int(0.25 * SAMPLE_RATE))
            pieces = []
            speaker = params["voice_dir"] + params["speaker"]
            # print(speaker)
            for i, sent in enumerate(sentences):
                audio_array = generate_audio(sent, history_prompt=speaker, text_temp=temp, waveform_temp=wave_temp)
                pieces += [audio_array, silence.copy()]
            outputs = np.concatenate(pieces)
            print(f"[BARK] wav: {outputs}")

        self.save_audio(outputs, output_fn, True)
            
        return outputs
    
    def save_audio(
        self,
        audio_output: torch.Tensor,
        filename: str,
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
        path = os.path.join(self.op_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not overwrite:
            print(f"File {path} already exists.")
            return

        print(f'Saving {filename} to {self.op_dir}')
        op = Audio(audio_output, rate=self.params[self.backend]["sample_rate"])
        print(op)
        with open(path+'.wav', 'wb') as f:
            f.write(op.data)

if __name__ == "__main__":
    voice = Speaker()
    text = "1. Gas chromatography is a separation technique that separates compounds in a gaseous mixture based on their physical properties. It is used in the field of chemical analysis to separate and identify various elements present in a sample. The process involves passing a gas through a small column containing a stationary phase. The stationary phase is usually packed into the column to separate the molecules. The molecules then interact with each other and the stationary phase, which leads to a separation and identification of individual components in the mixture."
    voice.say(text)