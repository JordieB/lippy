from dotenv import load_dotenv
from pathlib import Path

from lippy.utils.listener_via_api import AudioSplitter, Listener

# Set the filename for the input audio file to split and transcribe
AUDIO_FN = 'audio.mp3'

# Load the .env file (where your OpenAI API is stored in `OPENAI_API_KEY`)
_ = load_dotenv()

# Set file paths
cwd = Path.cwd()
data_dir = cwd/'data'
audio_dir = data_dir/'audio'
text_dir = data_dir/'text'
audio_fp = audio_dir/AUDIO_FN
output_fn = ''.join(AUDIO_FN.lower().replace(' ', '_').split('.')[:-1])
output_fn += '.txt'
OUTPUT_FILE_PATH = text_dir/output_fn
# Set how big the audio file segmenets will become
target_size_mb = 25

# Init mp3/speech splitter
splitter = AudioSplitter(input_file_path=audio_fp,
                        target_size_mb=target_size_mb,
                        output_file_path=OUTPUT_FILE_PATH)
# Split the MP3 by size and save the segments
output_file_paths = splitter.save_segments_to_directory()

# Transcribe the MP3 files via Whisper and ask GPT-4 to clean it up
SYSTEM_PROMPT = """
You are a helpful assistant for content speech-to-text transcription. Your 
task is to correct any spelling discrepancies in the transcribed text. 
Only add necessary punctuation such as periods, commas, and
capitalization, and use only the context provided.
"""
AUDIO_FILE_PATHS = output_file_paths

# Init transcriber
listener = Listener(SYSTEM_PROMPT)
# Run transcriber w/ corrector
listener.process_audio_files(AUDIO_FILE_PATHS, OUTPUT_FILE_PATH)
