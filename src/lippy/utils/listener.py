from pathlib import Path
import whisper

class Listener:
    def __init__(self):
        self.model = whisper.load_model('base')
    
    def transcribe(self, pathWav):
        print(f"Transcribing {pathWav.split('/')[-1]}")
        return self.model.transcribe(pathWav)


if __name__ == "__main__":
    PROJ_DIR = Path(__file__).resolve().parents[3]
    wav_fp = str(PROJ_DIR / "examples/Bark/wit_mem_passion.wav")

    listener = Listener()
    print(listener.transcribe(wav_fp)['text'])