import os
from pathlib import Path

from soundfile import write as sfwrite
from rouge_score import rouge_scorer
import torch
from nltk import sent_tokenize, download, Text
import numpy as np
from bark import generate_audio, SAMPLE_RATE, preload_models, save_as_prompt

from lippy.utils.listener import Listener

# Define project directory
PROJ_DIR = Path(__file__).resolve().parents[2]

class Speaker:
    def __init__(
        self,
        backend="bark",
        speaker_id="Dalinar-1_t8_w8_7",
        op_dir=PROJ_DIR / "data/audio",
        custom_voice_dir=PROJ_DIR / "data/voices",
        # TODO: fix this so it's either (a) OS-independent and/or (b) only uses
        #   PROJ_DIR
        # TODO: make PEP8 compliant ... somehow
        voice_dir=Path.home() / Path((".local/lib/python3.8/site-packages/bark"
                                      "/assets/prompts/v2/"))
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
    
    def split_runon(self, sentence):
        split_sentence = sentence.strip().split()
        line = ''
        split_len = 225
        wrapped_sentence = []
        for word in split_sentence:
            if len(line+word)<split_len:
                line += word + ' '
            else:
                wrapped_sentence.append(line)
                line = word + ' '
        wrapped_sentence.append(line)
        return wrapped_sentence

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
            pathPieces = []
            speaker = str(params["voice_dir"] / params["speaker"])
            for i, sent in enumerate(sentences):
                tries = 0
                if len(Text(sent)) >= 225:
                    sent = self.split_runon(sent)
                else: sent = [sent]
                for k, sents in enumerate(sent):
                    piece_fn = fnOutput + f"_p{i}_{k}"
                    # Generate tensor for sentence
                    full_gen, audio_array = generate_audio(sents,
                                                history_prompt=speaker,
                                                text_temp=temp,
                                                waveform_temp=wave_temp,
                                                output_full=True)

                    # # Save sentence clip
                    _ = self.save_audio(audio_array, piece_fn, True)
                    # Transcribe w/ whisper
                    _path = str(self.op_dir / Path(piece_fn + ".wav"))
                    promptTranscript = self.listener.transcribe(_path)
                    # Score recall using RougeL scorer
                    score = self.rouge.score(
                        sents,
                        promptTranscript['text'])['rougeL'][2]
                    print(f"---\n"
                          f"Part {i}, try {tries}:\n"
                          f"Original:   {sents.strip()}\n"
                          f"Transcript: {promptTranscript['text'].strip()}\n"
                          f"F1:         {score}")
                    # If recall < acceptable value, regen audio
                    while score < .9 and tries <= 3:
                        full_gen, audio_array = generate_audio(sents,
                                                history_prompt=speaker,
                                                text_temp=temp,
                                                waveform_temp=wave_temp,
                                                output_full=True)
                        _ = self.save_audio(audio_array, piece_fn, True)
                        tries += 1
                        _path = str(self.op_dir / Path(piece_fn + ".wav"))
                        promptTranscript = self.listener.transcribe(_path)
                        # Score recall using RougeL scorer
                        score = self.rouge.score(sents,
                                                 promptTranscript['text'])
                        score = score['rougeL'][2]
                        print(f"---\n"
                              f"Part {i}, try {tries}:\n"
                              f"Original:   {sents.strip()}\n"
                              f"Transcript: {promptTranscript['text'].strip()}"
                              f"\nF1:         {score}")
                    # append to pieces w/ silence
                    speaker_path = {params['voice_dir'] / params['speaker']}
                    speaker_path += f'_{i}_{k}.npz'
                    pathPieces.append(speaker_path)
                    save_as_prompt(speaker, full_gen)
                    pieces += [audio_array, silence.copy()]
                    pathPieces.append(self.op_dir / Path(piece_fn +'.wav'))
            # Merge pieces to single chunk
            outputs = np.concatenate(pieces)
            for path in pathPieces:
                os.remove(path)
            print(f"[BARK] wav: {outputs}")
        # Save chunk as wav
        _ = self.save_audio(outputs, fnOutput, True)
        return outputs
    
    def save_audio(self, audio_output, filename, overwrite=False):
        """
        Saves the audio as a WAV file.

        Args:
            audio_output (numpy.ndarray): The audio to be saved as a numpy array.
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
        sf.write(path + '.wav', audio_output,
                 self.params[self.backend]["sample_rate"])

# Main module check
if __name__ == "__main__":
    voice = Speaker()
    text = ("And so I stand among you as one that offers a small message of "
            "hope, that first, there are always people who dare to seek on the"
            " margin of society, who are not dependent on social acceptance, "
            "not dependent on social routine, and prefer a kind of "
            "free-floating existence.")
    voice.say(text)
