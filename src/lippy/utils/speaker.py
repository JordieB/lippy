import os
import torch
from scipy.io import wavfile

class Speaker:
    def __init__(self, base_dir: str = "audio_files"):
        """
        Initializes the Speaker class.

        Parameters:
        base_dir (str, optional): The base directory where the audio files will
                                  be saved.
                                  Defaults to "audio_files".
        """
        self.lang = 'en'
        self.model_id = 'v3_en'
        self.samp = 48000
        self.speaker = 'en_0'
        self.device = torch.device('cpu')
        self.base_dir = base_dir
        self.load_model()

    def load_model(self):
        """
        Loads the pre-trained TTS model.
        """
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=self.lang,
            speaker=self.model_id
        )
        self.model.to(self.device)

    def say(self, input: str) -> torch.Tensor:
        """
        Generates speech from the input text.

        Parameters:
        input (str): The text to be spoken.

        Returns:
        torch.Tensor: The generated speech audio as a tensor.
        """
        return self.model.apply_tts(
            text=input,
            speaker=self.speaker,
            sample_rate=self.samp
        )

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
        path = os.path.join(self.base_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not overwrite:
            print(f"File {path} already exists.")
            return

        print(f'Saving {filename} to {self.base_dir}')
        wavfile.write(path, self.samp, audio_output.detach().numpy())
