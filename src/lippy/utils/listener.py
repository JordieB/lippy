import whisper

class Listener:
    def __init__(self):
        self.model = whisper.load_model('base')
    
    def transcribe(self, pathWav):
        print(f"Transcribing {pathWav.split('/')[-1]}")
        return self.model.transcribe(pathWav)


if __name__ == "__main__":
    wav_fp = "/home/ubuntu/Tehas/lippy/example/Bark/wit_mem_passion.wav"

    listener = Listener()
    print(listener.transcribe(wav_fp)['text'])