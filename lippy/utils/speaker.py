import os
import torch
import wave
from nltk import sent_tokenize, download
import numpy as np
from bark import generate_audio, SAMPLE_RATE, preload_models
from pathlib import Path
from lippy.utils.listener import Listener
from rouge_score import rouge_scorer
from IPython.display import Audio

PROJ_DIR = Path(__file__).resolve().parents[2]

class Speaker:
    def __init__(
        self,
        backend="bark",
        speaker_id="Dalinar-1_t8_w8_7",
        op_dir=PROJ_DIR / "data/audio",
        custom_voice_dir=PROJ_DIR / "voices",
        # TODO: fix this so it's either (a) OS-independent and/or (b) only uses
        #   PROJ_DIR
        # TODO: make PEP8 compliant ... somehow
        voice_dir=Path.home() / Path(".local/lib/python3.8/site-packages/bark/assets/prompts/v2/")
    ):
        """
        Initializes the Speaker class.

        Args:
            backend (str): The backend to use for text-to-speech. Defaults to
                "bark".
            speaker_id (str): The ID of the speaker voice to use. Defaults to
                "Dalinar-1_t8_w8_7".
            op_dir (Path): The directory where the audio files will be saved.
            custom_voice_dir (Path): The directory containing custom voice
                files.
            voice_dir (Path): The directory containing the default voice files.
        """
        self.backend = backend
        self.params = {
            "silero": {
                "lang": "en",
                "model_id": "v3_en",
                "sample_rate": 48000,
                "speaker": speaker_id,
                "device": torch.device("cpu"),
            },
            "bark": {
                "speaker": speaker_id + ".npz",
                "sample_rate": SAMPLE_RATE,
                "custom_voice_dir": custom_voice_dir,
                "voice_dir": custom_voice_dir
                if (custom_voice_dir / Path(speaker_id + ".npz")).exists()
                else voice_dir,
            },
        }
        self.op_dir = Path(op_dir)
        self.load_model()
        self.listener = Listener()
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def load_model(self):
        """
        Loads the pre-trained TTS model based on the backend specified during
        initialization.
        """
        params = self.params[self.backend]
        if self.backend == "silero":
            self.model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language=params["lang"],
                speaker=params["speaker"],
            )
            self.model.to(self.device)
        elif self.backend == "bark":
            download("punkt")
            preload_models()
        
    def split_into_sentences(self, input_string):
        """
        Splits the input string into individual sentences.

        Args:
            input_string (str): The string to split into sentences.

        Returns:
            list: A list of sentences.
        """
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

    def say(self, input:str, fnOutput="output", temp=0.8, wave_temp=0.8):
        """
        Converts the input text into speech.

        Args:
            input (str): The text to convert into speech.
            fnOutput (str): The filename for the output audio file.
            temp (float): The temperature for the text generation.
                Only used for the "bark" backend.
            wave_temp (float): The temperature for the waveform generation.
                Only used for the "bark" backend.

        Returns:
            torch.Tensor: The audio output.
        """
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
            speaker = str(params["voice_dir"] / params["speaker"])
            for i, sent in enumerate(sentences):
                piece_fn = fnOutput + f"_p{i}"
                # Generate tensor for sentence
                audio_array = generate_audio(sent,
                                             history_prompt=speaker,
                                             text_temp=temp,
                                             waveform_temp=wave_temp)

                # # Save sentence clip
                _ = self.save_audio(audio_array, piece_fn, True)

                # Transcribe w/ whisper
                promptTranscript = self.listener.transcribe(str(self.op_dir / Path(piece_fn + ".wav")))
                # Score recall using RougeL scorer
                score = self.rouge.score(sent, promptTranscript['text'])['rougeL'][2]
                print(f"""---
Part {i}:
  Original:   {sent.strip()}
  Transcript: {promptTranscript['text'].strip()}
  F1:         {score}""")
                # If recall < .95, regen audio
                if score < .95:
                    audio_array = generate_audio(sent,
                                             history_prompt=speaker,
                                             text_temp=temp,
                                             waveform_temp=wave_temp)
                # append to pieces w/ silence
                os.remove(self.op_dir / Path(piece_fn +'.wav'))
                pieces += [audio_array, silence.copy()]
            # Merge pieces to single chunk
            outputs = np.concatenate(pieces)

            print(f"[BARK] wav: {outputs}")
        # Save chunk as wav
        _ = self.save_audio(outputs, fnOutput, True)
        return outputs
    
    def save_audio(
        self,
        audio_output: torch.Tensor,
        filename: str,
        overwrite: bool = False
    ):
        """
        Saves the audio as a WAV file.

        Args:
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
        with open(path+'.wav', 'wb') as f:
            f.write(op.data)

if __name__ == "__main__":
    voice = Speaker()
    text = ("1. Gas chromatography is a separation technique that separates "
            "compounds in a gaseous mixture based on their physical properties."
            " It is used in the field of chemical analysis to separate and "
            "identify various elements present in a sample. The process "
            "involves passing a gas through a small column containing a "
            "stationary phase. The stationary phase is usually packed into the "
            "column to separate the molecules. The molecules then interact with"
            " each other and the stationary phase, which leads to a separation "
            "and identification of individual components in the mixture.")
    voice.say(text)
