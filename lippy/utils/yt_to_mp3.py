import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
from pytube import Playlist, YouTube


class YouTubeToMP3:
    """A class to download YouTube videos and convert them to MP3 format."""

    def __init__(self, output_path: Path) -> None:
        """Initialize the class with the path where outputs will be saved.

        Args:
            output_path (Path): The output directory for saved files.
        """
        self.output_path = output_path
        self.logs = []

    def convert_mp4_to_mp3(self, mp4_path: str) -> str:
        """Convert an MP4 file to MP3 format using ffmpeg.

        Args:
            mp4_path (str): The file path of the MP4 file to convert.

        Returns:
            str: The file path of the converted MP3 file.
        """
        start_time = time.time()

        # Extract filename without extension for the new MP3 file
        base_fn = mp4_path.stem
        mp3_path = mp4_path.with_suffix('.mp3')
        
        # Command to convert MP4 to MP3 using ffmpeg
        subprocess.run([
            'ffmpeg', '-i', str(mp4_path), '-q:a', '0', '-map', 'a', '-y',
            str(mp3_path)
        ], check=True)
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f'Converted in {elapsed:.2f} secs - {base_fn}')

        # Update logs
        log = {
            'step': 'convert',
            'obj': str(mp3_path),
            'elapsed_sec': elapsed
        }
        self.logs.append(log)

        return mp3_path
    
    def create_list_of_vids(self, url: str) -> str:
        """Create a list of YouTube objects to use for downloading videos.

        Args:
            url (str): The URL of the YouTube video or playlist.

        Returns:
            List: a list of pytube.YouTube objects to download
        """

        # Establish which videos will be downloaded
        if 'playlist' in url:
            playlist = Playlist(url)
            videos = playlist.videos
        else:
            video = YouTube(url)
            videos = [video]
        
        return videos

    def download_yt(self, videos: List[YouTube]) -> str:
        """Download a video from a YouTube URL.

        Args:
            url (str): The URL of the YouTube video or playlist.

        Returns:
            str: The file path of the downloaded MP4 video.
        """

        mp4_paths = []
        
        # Download video(s)
        for video in videos:
            start_time = time.time()
            mp4_path = video.streams.get_highest_resolution().download(
                output_path=str(self.output_path))
            end_time = time.time()
            elapsed = end_time - start_time
            print((f'Download in {elapsed:.2f} secs - '
                   f'{video.title}'))
            mp4_paths.append(Path(mp4_path))

        # Update logs
        log = {
            'step': 'download',
            'obj': mp4_path,
            'elapsed_sec': elapsed
        }
        self.logs.append(log)

        return mp4_paths

    def process_url(self, url: str) -> Tuple[str, str]:
        """Download a YouTube video and convert it to MP3.

        Args:
            url (str): The URL of the YouTube video or playlist.

        Returns:
            Tuple[str, str]: A tuple containing the paths of the MP4 and MP3 files.
        """
        mp4_paths = self.download_yt(url)
        mp3_paths = []
        for mp4_path in mp4_paths:
            mp3_path = self.convert_mp4_to_mp3(mp4_path)
            mp3_paths.append(mp3_path)

        return mp4_paths, mp3_paths


if __name__ == '__main__':
    # Define directories and create them if they don't exist
    data_dir = Path('data')
    speech_dir = data_dir / 'speech'
    logs_dir = data_dir/'logs'
    speech_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    logs_path = logs_dir/'speech-to-text_pipeline_logs.parquet'


    # YouTube video and playlist URLs to download and convert
    videos_and_playlists = [
        # Dazs - How To Get Better At Apex Legends VOD Review
        ('https://www.youtube.com/playlist?list=PL_waWDJmtQZN0tg3zU4z1QwZu-Ih'
         '8lC2k'),
        # 2023 Aim Guide To Improve Your Aim on Apex Legends (Aim Categories & Self Improvement Tips)
        'https://youtu.be/2evMaU5uvAM?feature=shared'
    ]

    # Init list to collect results
    unflattened_results = []

    # Initialize the converter and process each URL
    dl_converter = YouTubeToMP3(speech_dir)
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

