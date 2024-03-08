# Lippy

## Project Description

A local LLM-based AI assistant built in Python with the help of LangChain, OpenAI (Whisper, GPT), Bark, and Chroma.

## Features

### MVP
- Create data for RAG from:
    - YouTube Videos
    - MP3 files
    - Obsidian/Markdown files
- TTS for ebooks

### Planned
- Retrieve upcoming events from Google Calendar
- Use AI language model (GPT) to find the next meeting
- Object-oriented design with proper documentation and linting
- Environment variable configuration for sensitive information

## Installation

This installation method is designed to be user-friendly, assuming you have Python, Git, and ffmpeg already installed on your system. The setup script automates the process of cloning the repository, creating a Python virtual environment, installing necessary Python packages, and ensuring ffmpeg is installed for audio file handling.

### Prerequisites

Before running the installation command, please ensure the following requirements are met:

- **Python 3.x**: Ensure Python 3 is installed on your system. You can verify this by running `python3 --version` in your terminal. If you need to install Python, visit [the official Python website](https://www.python.org/downloads/) for download instructions.
- **Git**: Git is required to clone the repository. Verify its installation by running `git --version`. Visit [Git's official site](https://git-scm.com/downloads) for installation instructions if needed.
- **ffmpeg**: For handling audio files, ffmpeg needs to be installed. You can check if ffmpeg is installed by running `ffmpeg -version`. For installation instructions, refer to [ffmpeg's official website](https://ffmpeg.org/download.html).

### Installation Command

Once you have the prerequisites installed, you can set up the project using the following command:

```shell
curl https://raw.githubusercontent.com/JordieB/lippy/main/project_setup.py | python3
```

### What the Script Does
The setup script performs the following actions:

1. **Clone the Repository**: It clones the `lippy` project from GitHub to your local machine.
2. **Create a Virtual Environment**: A Python virtual environment is created within the cloned project directory. This isolated environment ensures that the project's dependencies do not interfere with the system-wide Python packages.
3. **Install Python Dependencies**: The script installs all necessary Python packages listed in `requirements.txt` into the virtual environment. It also installs the project in editable mode (`pip install -e .`), which is useful for development purposes.
4. Install `ffmpeg` (for Unix/Linux/MacOS users): If you're on a Unix/Linux/MacOS system, the script attempts to install `ffmpeg` using the system's package manager (e.g., `apt-get` for Debian/Ubuntu). Windows users will need to install ffmpeg manually if it's not already installed.

#### Note for Windows Users

Windows users need to ensure ffmpeg is installed manually. The recommended method is to use a package manager like Chocolatey (`choco install ffmpeg`). Alternatively, you can download ffmpeg from its official website and follow the installation instructions.

### Final Steps
After running the installation command, you'll have a fully set up project environment ready for development or use. For further instructions on how to use the project, refer to the subsequent sections of this README.

## Usage

Under Construction

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
