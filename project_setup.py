import os
import subprocess
import sys
import venv

# Function to run a command and print its output
def run_command(cmd, cwd=None):
    """
    Runs a command and prints its output.

    Args:
        cmd (List[str]): The command to be executed, represented as a list of strings.
        cwd (str): The current working directory to execute the command in.

    Raises:
        SystemExit: If the command returns a non-zero exit code, indicating an error.
    """
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {cmd}")
        print(stderr.decode())
        sys.exit(1)
    print(stdout.decode())

# Clone the git repo
run_command(["git", "clone", "https://github.com/JordieB/lippy.git"])

# Move to the cloned repo
os.chdir("lippy")

# Create a virtual environment
venv_dir = "venv"
venv.create(venv_dir, with_pip=True)

# Depending on the OS, the path to the Python interpreter in the virtual environment will be different
if os.name == "nt":  # For Windows
    python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
else:  # For Unix or MacOS
    python_executable = os.path.join(venv_dir, "bin", "python")

# Set-up pip commands to handle libs and editable mode for project
pip_install_commands = [
    [python_executable, "-m", "pip", "install", "-r", "requirements.txt"],
    [python_executable, "-m", "pip", "install", "-e", "."],
    # Install Bark for text-to-voice generation
    [python_executable, "-m", "pip", "install", "git+https://github.com/suno-ai/bark.git"],
]

# Run each pip install command
for command in pip_install_commands:
    run_command(command)

# Install ffmpeg
if os.name == "nt":  # For Windows
    print(
        ("Please install ffmpeg manually. Recommendation: use a package manager like "
         "Chocolatey (`choco install ffmpeg`).")
    )
else:  # For Unix or MacOS
    print('Installing ffmpeg for pydub/audio file handling for project...')
    run_command(["sudo", "apt-get", "install", "-y", "ffmpeg"])
