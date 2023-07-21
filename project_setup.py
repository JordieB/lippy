import os
import subprocess
import sys
import venv

# Function to run a command and print its output
def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {cmd}")
        print(stderr.decode())
        sys.exit(1)
    print(stdout.decode())

# Clone the git repo
run_command(["git", "clone", "https://github.com/JordieB/lippy.git"])

# Create a virtual environment
venv_dir = "env"
venv.create(venv_dir, with_pip=True)

# Activate the virtual environment and install dependencies
pip_install_commands = [
    ["pip", "install", "-r", "requirements.txt"],
    ["pip", "install", "-e", "."],
    # Install Bark for text-to-voice generation
    ["pip", "install", "git+https://github.com/suno-ai/bark.git"],
]

# Depending on the OS, the path to the Python interpreter in the virtual environment will be different
if os.name == "nt":  # For Windows
    python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
else:  # For Unix or MacOS
    python_executable = os.path.join(venv_dir, "bin", "python")

# Run each pip install command
for command in pip_install_commands:
    run_command([python_executable, "-m"] + command)

# Install ffmpeg
if os.name == "nt":  # For Windows
    print("Please install ffmpeg manually or use a package manager like Chocolatey (choco install ffmpeg).")
else:  # For Unix or MacOS
    run_command(["sudo", "apt-get", "install", "ffmpeg"])
