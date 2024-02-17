# TODO: add logs to AudioSplitter and Listener

from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from lippy.utils.misc import find_project_root
from lippy.utils.listener_via_api import AudioSplitter, Listener
from lippy.utils.yt_to_mp3 import YouTubeToMP3

# Load the .env file (where your OpenAI API is stored in `OPENAI_API_KEY`)
_ = load_dotenv()

# Define directories and create them if they don't exist
project_root = Path(find_project_root())
data_dir = project_root/'data'
video_dir = data_dir/'video'
video_dir.mkdir(parents=True, exist_ok=True)
audio_dir = data_dir/'audio'
audio_dir.mkdir(parents=True, exist_ok=True)
text_dir = data_dir/'text'
text_dir.mkdir(parents=True, exist_ok=True)
logs_dir = data_dir/'logs'
logs_dir.mkdir(parents=True, exist_ok=True)

# Create specific input/output file paths
logs_path = logs_dir/'speech-to-text_pipeline_logs.parquet'

# YouTube video and playlist URLs to download and convert
videos_and_playlists = [
    # (Playlist) Dazs - How To Get Better At Apex Legends VOD Review
    ('https://www.youtube.com/playlist?list=PL_waWDJmtQZN0tg3zU4z1QwZu'
     '-Ih8lC2k'),
    # (Video) Dazs -2023 Aim Guide To Improve Your Aim on Apex Legends (Aim 
    # Categories & Self Improvement Tips)
    'https://youtu.be/2evMaU5uvAM?feature=shared'
]

# Init list to collect results
unflattened_results = []

# Initialize the YT downloader + coverter
dl_converter = YouTubeToMP3(
    mp4_dir=video_dir,
    mp3_dir=audio_dir
)
# Download YT as MP4 and convert them to MP3
for url in videos_and_playlists:
    mp4_paths, mp3_paths = dl_converter.process_url(url)
    unflattened_results.append(mp4_paths)
    unflattened_results.append(mp3_paths)
    print(f'Downloaded and converted: {url}')

# NOTE: `mixed_lists` could have single Path objs w/ lists of Path objs
# Ensure all Path objs are flattened into single list
results = []
for item in unflattened_results:
    # If element is a list of Path objs
    if isinstance(item, list):
        # Extend final list to flatten
        results.extend(item)
    else:
        # Otherwise just append the single Path obj to keep flat
        results.append(item)

# Store logs
logs = pd.DataFrame(dl_converter.logs)
logs.to_parquet(logs_path,
                engine='pyarrow',
                index=False,
                compression='snappy',
                mode='append')

# Just get MP3 paths
mp3_paths = [item for item in results if ''.join(item.suffixes) == '.mp3']
# NOTE: AudioSplitter will break MP3s down to smaller files if bigger than
# desired, useful for ensure Whisper can process the MP3
target_size_mb = 25

for mp3_path in mp3_paths:
    base_fn = mp3_path.stem
    base_txt_path = base_fn.with_suffix('.txt')
    # Init mp3 splitter
    splitter = AudioSplitter(target_size_mb=target_size_mb,
                             input_file_path=mp3_path,
                             output_file_path=mp3_path)
    # Split the MP3 by size and save the segments
    output_file_paths = splitter.save_segments_to_directory()
    # Transcribe the MP3 files via Whisper and ask GPT-4 to clean it up
    system_prompt = """
    You are a helpful assistant for content speech-to-text transcription. Your 
    task is to correct any spelling discrepancies in the transcribed text. 
    Only add necessary punctuation such as periods, commas, and
    capitalization, and use only the context provided.
    """
    # Init transcriber w/ corrector
    listener = Listener(system_prompt)
    # Run transcriber w/ corrector
    listener.process_audio_files(output_file_paths, base_txt_path)
