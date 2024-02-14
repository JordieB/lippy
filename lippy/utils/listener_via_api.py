import os
from typing import List

from openai import ChatCompletion, Audio
from pydub import AudioSegment

class AudioSplitter:
    """
    A class to split audio files into smaller segments based on target size.

    Attributes:
        input_file_path (str): Path to the input audio file.
        target_size_mb (float): Desired size (in MB) of the output segments.
        output_file_path (str): Directory where the segments will be saved.
    """

    def __init__(self, input_file_path: str, target_size_mb: float, 
                 output_file_path: str):
        """
        Initializes the AudioSplitter class with the given parameters.

        Args:
            input_file_path (str): Path to the input audio file.
            target_size_mb (float): Desired size (in MB) of the output segments.
            output_file_path (str): Directory where the segments will be saved.
        """
        self.input_file_path = input_file_path
        self.target_size_mb = target_size_mb
        self.output_file_path = output_file_path

    def estimate_duration_for_size(self, audio_length_ms: int,
                                   audio_file_size_mb: float) -> int:
        """
        Estimates the duration in milliseconds for the desired segment size.

        Args:
            audio_length_ms (int): Total length of the audio in milliseconds.
            audio_file_size_mb (float): Size of the audio file in MB.

        Returns:
            int: Estimated duration in milliseconds for the segment.

        Example usage:
            # Set file path
            mp3_input_file_path = 'path/to/audio.mp3'
            # Load in audio
            audio = AudioSegment.from_mp3(mp3_input_file_path)
            # Est numbers to calc estimated duratio of each audio segment
            audio_length_ms = len(audio)
            audio_file_size_mb = os.path.getsize(mp3_input_file_path)
            audio_file_size_mb = audio_file_size_mb / 1024**2
            # Run calc to estimate duration of each audio segment in ms
            estimated_duration_ms = self.estimate_duration_for_size(
                audio_length_ms=audio_length_ms,
                audio_file_size_mb=audio_file_size_mb
            )

        """
        return int(
            (self.target_size_mb / audio_file_size_mb)
            * audio_length_ms
        )

    def _split_audio_to_size_limit(self) -> List[AudioSegment]:
        """
        Splits the audio into segments based on the target size.

        Returns:
            List[AudioSegment]: List of audio segments.
        """
        # Load initial audio file
        audio = AudioSegment.from_mp3(self.input_file_path)
        # Estimate duration of each audio segment in ms
        audio_length_ms = len(audio)
        audio_file_size_mb = os.path.getsize(self.input_file_path)
        audio_file_size_mb = audio_file_size_mb / 1024**2
        estimated_duration_ms = self.estimate_duration_for_size(
            audio_length_ms, audio_file_size_mb
        )

        # Establish some tracking data to help managing looping operations
        segments = []
        start_time = 0

        # Segment out audio file less than or equal to the split length (aka
        # estimated duration in ms)
        while start_time < len(audio):
            end_time = start_time + estimated_duration_ms
            segment = audio[start_time:end_time]

            tmp_fp = 'tmp/interim_segment.mp3'
            segment.export(tmp_fp, format="mp3")

            while ((os.path.getsize(tmp_fp)
                        > (self.target_size_mb * 1024 * 1024))
                   and (estimated_duration_ms > 0)):
                estimated_duration_ms -= 1000
                segment = audio[start_time:start_time + estimated_duration_ms]
                segment.export(tmp_fp, format="mp3")

            segments.append(segment)

            start_time += estimated_duration_ms

        os.remove(tmp_fp)

        return segments

    def save_segments_to_directory(self) -> List[str]:
        """
        Saves the audio segments based on the specified output_file_path path.

        The segment files will have names based on the last part of the
        output_file_path path (without extension), appended with segment
        indices.
        
        If output_file_path looks like a full file path:
            "path/to/file_name.mp3", 
        the files will be saved as:
            "path/to/file_name_0.mp3", "path/to/file_name_1.mp3", ...

        Returns:
            List[str]: List of paths to the saved segment files.

        Example usage:
            input_file_path = "good_morning.mp3"
            target_size_mb = 25
            output_base_fn = "output_segments"
            splitter = AudioSplitter(input_file_path, target_size_mb,
                                     output_base_fn)
            output_file_paths = splitter.save_segments_to_directory()
        """
        segments = self._split_audio_to_size_limit()

        directory_path, full_filename = os.path.split(self.output_file_path)
        base_name, _ = os.path.splitext(full_filename)

        # Ensure the directory exists
        os.makedirs(directory_path, exist_ok=True)

        output_files = []

        for i, segment in enumerate(segments):
            output_file = os.path.join(directory_path, f"{base_name}_{i}.mp3")
            segment.export(output_file, format="mp3")
            output_files.append(output_file)

        return output_files

class Listener:
    """
    Transcribes audio using OpenAI's Whisper model and passes the 
    transcription through their GPT-4 model for LLM-assisted correction and 
    cleaning.
    """

    def __init__(self, system_prompt: str):
        """
        Initializes the TranscriptCorrector with a system prompt.

        Args:
            system_prompt (str): System prompt for the GPT-4 model.
        """
        self.system_prompt = system_prompt

    def transcribe_audio(self, audio_file_path: str) -> str:
        """
        Utilizes the Whisper model to transcribe the audio file.
        
        Args:
            audio_file_path (str): Path to the audio file.

        Returns:
            (str): transcription generated via OpenAI's API for Whisper

        Example Usage:
            SYSTEM_PROMPT = 'test system prompt, please ignore'
            AUDIO_FILE_PATH = 'path/to/audio.mp3'
            listener = Listener(SYSTEM_PROMPT)
            transcription = listener.transcribe_audio(AUDIO_FILE_PATH)
        """
        with open(audio_file_path, 'rb') as audio_file:
            return Audio.transcribe(model='whisper-1', file=audio_file)['text']

    def generate_corrected_transcript(self, transcription: str) -> str:
        """
        Generates a corrected transcript for a given audio file.

        Args:
            transcription (str): transcription to improve, provided via
                self.transcribe_audio()
            temperature (float, optional): Sampling temperature for GPT-4. 
                Defaults to 0.0.

        Returns:
            str: LLM-assisted clean and corrected transcription

        Example usage:
            SYSTEM_PROMPT = 'test system prompt, please ignore'
            AUDIO_FILE_PATH = 'path/to/audio.mp3'
            listener = Listener(SYSTEM_PROMPT)
            transcription = listener.transcribe_audio(AUDIO_FILE_PATH)
            improved_transcription = listener \
                .generate_corrected_transcript(transcription)
        """
        response = ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": transcription
                }
            ]
        )
        response_str = response['choices'][0]['message']['content']
        return response_str

    def process_audio_files(self, audio_file_paths: List[str], 
                            output_file_path: str):
        """
        Processes a list of audio files and saves the corrected transcripts 
        to an output file.

        Args:
            audio_file_paths (List[str]): List of paths to the audio files.
            output_file_path (str): Path to the output text file.

        Example usage:
            SYSTEM_PROMPT = ("You are a helpful assistant for content ",
                             "speech-to-text transcription. Your task is to ",
                             "correct any spelling discrepancies in the ",
                             "transcribed text. Only add necessary ",
                             "punctuation such as periods, commas, and ",
                             "capitalization, and use only the context ",
                             "provided.")
            AUDIO_FILE_PATHS = [
                'data/speech/no_time_to_make_social_media_content_0.mp3',
                'data/speech/no_time_to_make_social_media_content_1.mp3',
                'data/speech/no_time_to_make_social_media_content_3.mp3'
            ]
            OUTPUT_FILE_PATH = ('data/text/',
                                'no_time_to_make_social_media_content.txt')

            # Init Listener
            listener = Listener(SYSTEM_PROMPT)
            # Transcribe via Listener
            listener.process_audio_files(AUDIO_FILE_PATHS, OUTPUT_FILE_PATH)
        """
        with open(output_file_path, 'a') as output_file:
            for audio_file_path in audio_file_paths:
                transcription = self.transcribe_audio(audio_file_path)
                corrected_text = self.generate_corrected_transcript(
                    transcription,
                    audio_file_path
                )
                output_file.write(corrected_text)

if __name__ == '__main__':
    from dotenv import load_dotenv
    from pathlib import Path

    from lippy.utils.misc import find_project_root
    
    # Set the filename for the input audio file to split and transcribe
    audio_fn = 'audio.mp3'

    # Load the .env file (where your OpenAI API is stored in `OPENAI_API_KEY`)
    _ = load_dotenv()

    # Set file paths
    project_root_fp = find_project_root()
    data_dir = project_root_fp/'data'
    audio_dir = data_dir/'audio'
    text_dir = data_dir/'text'
    INPUT_AUDIO_FP = audio_dir/audio_fn
    output_fn = ''.join(audio_fn.lower().replace(' ', '_').split('.')[:-1])
    output_fn += '.txt'
    OUTPUT_TEXT_FP = text_dir/output_fn
    # Set how big the audio file segmenets will become
    target_size_mb = 25

    # Set how big the audio file segmenets will become
    target_size_mb = 25
    
    # Init mp3/speech splitter
    splitter = AudioSplitter(input_file_path=INPUT_AUDIO_FP,
                            target_size_mb=target_size_mb,
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
