# Lippy

A Python application to retrieve upcoming events from Google Calendar and use an AI language model to find the next meeting.

## Description

This project uses the `gcalwrapper` module to interact with the Google Calendar API and retrieve upcoming events. It then employs the `langchain` package to leverage an AI language model (GPT) to generate a response with details about the next meeting.

## Features

- Retrieve upcoming events from Google Calendar
- Use AI language model (GPT) to find the next meeting
- Object-oriented design with proper documentation and linting
- Environment variable configuration for sensitive information

## Installation

More user-friendly, requires Python and Git w/ ffmpeg (preferrably) already installed:
```shell
curl https://raw.githubusercontent.com/JordieB/lippy/main/project_setup.py | python3
```

Less user-friendly for Unix/MacOS (w/o using virtual enviornments):
```shell
git clone https://github.com/JordieB/lippy.git
cd lippy
pip install -e .
pip install -r requrirements.txt
bash ./install_dependencies.sh
```

## Usage

Under Construction

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
