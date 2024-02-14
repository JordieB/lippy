from dotenv import load_dotenv
from pathlib import Path

from lippy.utils.misc import find_project_root
from lippy.utils.listener_via_api import AudioSplitter, Listener
    
# Set the filename for the input audio file to split and transcribe
audio_fn = 'audio.mp3'

# Load the .env file (where your OpenAI API is stored in `OPENAI_API_KEY`)
_ = load_dotenv()

# Set file names and paths
project_root_fp = find_project_root()
data_dir = project_root_fp/'data'
audio_dir = data_dir/'audio'
text_dir = data_dir/'text'
INPUT_AUDIO_FP = audio_dir/audio_fn
output_fn = ''.join(audio_fn.lower().replace(' ', '_').split('.')[:-1])
output_fn += '.txt'
OUTPUT_TEXT_FP = text_dir/output_fn
# Set how big the audio file segmenets will become
TARGET_SIZE_MB = 25

# Init mp3/speech splitter
splitter = AudioSplitter(input_file_path=INPUT_AUDIO_FP,
                        target_size_mb=TARGET_SIZE_MB,
                        # Will add numbers to base file name to denote 
                        # segments of the original mp3
                        output_file_path=INPUT_AUDIO_FP)
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
listener.process_audio_files(AUDIO_FILE_PATHS, OUTPUT_TEXT_FP)
