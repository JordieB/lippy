from pathlib import Path
from typing import Dict
import whisper


class Listener:
    def __init__(self, model_name: str = "base"):
        """
        Initializes the Listener class.

        Args:
            model_name (str): The name of the model to load. Defaults to "base".
        """
        self.model = whisper.load_model(model_name)

    def transcribe(self, path_wav: str) -> Dict[str, str]:
        """
        Transcribes the audio file at the specified path.

        Args:
            path_wav (str): The path to the audio file.

        Returns:
            Dict[str, str]: The transcription result.
        """
        # Extract the filename from the path for printing
        filename = Path(path_wav).name
        print(f"Transcribing {filename}")

        # Transcribe the audio file and return the result
        return self.model.transcribe(path_wav)


if __name__ == "__main__":
    # Define the project directory
    PROJ_DIR = Path(__file__).resolve().parents[3]

    # Define the path to the audio file
    wav_fp = str(PROJ_DIR / "examples/Bark/wit_mem_passion.wav")

    # Initialize the Listener
    listener = Listener()

    # Transcribe the audio file and print the result
    print(listener.transcribe(wav_fp)["text"])
