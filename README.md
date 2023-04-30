# Project Title

A Python application to retrieve upcoming events from Google Calendar and use an AI language model to find the next meeting.

## Description

This project uses the `gcalwrapper` module to interact with the Google Calendar API and retrieve upcoming events. It then employs the `langchain` package to leverage an AI language model (GPT) to generate a response with details about the next meeting.

## Features

- Retrieve upcoming events from Google Calendar
- Use AI language model (GPT) to find the next meeting
- Object-oriented design with proper documentation and linting
- Environment variable configuration for sensitive information

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JordieB/lippy.git
```
2. Change directory to the project folder:
```bash
cd lippy
```
3. Install the required dependencies using `pip` or `conda`:
```bash
pip install -r requirements.txt
```
4. Set up environment variables for your Google Calendar API credentials and calendar ID:
```bash
export GOOGLE_CAL_ID="your_calendar_id"
```
5. Replace the `service_file_path` variable in the code with the path to your Google Calendar API service file in PKL format.

## Usage

1. Run the main script:
```bash
python main.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)