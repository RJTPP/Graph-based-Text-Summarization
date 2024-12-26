import os
import subprocess
import sys
import threading
import time

def spinner(message):
    """Display a spinner animation."""
    spinner_chars = "|/-\\"
    idx = 0
    stop_event = threading.Event()

    def spin(sleep_time=0.25):
        nonlocal idx
        while not stop_event.is_set():
            print(f"\r{message} {spinner_chars[idx % len(spinner_chars)]}", end="")
            idx += 1
            time.sleep(sleep_time)
        print("\r" + " " * len(message) + "\r", end="")  # Clear the spinner line

    thread = threading.Thread(target=spin)
    thread.start()
    return stop_event

def run_with_spinner(command, message):
    """Run a subprocess command with a spinner animation."""
    stop_event = spinner(message)
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        stop_event.set()

def create_venv(venv_dir="venv"):
    """Create a virtual environment if it doesn't exist."""
    if os.path.exists(venv_dir):
        print(f"Virtual environment already exists in {venv_dir}.")
        if input("Do you want to overwrite it? (y/n): ").lower() != "y":
            return
        else:
            subprocess.run(["rmdir", "/s", "/q", venv_dir] if os.name == "nt" else ["rm", "-rf", venv_dir], check=True)

    run_with_spinner([sys.executable, "-m", "venv", venv_dir], f"Creating virtual environment in {venv_dir}...")
    print()


def upgrade_pip(venv_dir="venv"):
    """Upgrade pip in the virtual environment."""
    pip_executable = os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin", "pip")
    run_with_spinner([pip_executable, "install", "--upgrade", "pip"], "Upgrading pip...")
    print()


def install_requirements(requirements_file="requirements.txt", venv_dir="venv"):
    """Install required packages from requirements.txt."""
    pip_executable = os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin", "pip")
    run_with_spinner([pip_executable, "install", "-r", requirements_file], "Installing required packages...")
    print()


def create_dataset_folder(folder="dataset"):
    """Ensure the dataset folder exists."""
    print(f"Ensuring dataset folder exists: {folder}")
    os.makedirs(folder, exist_ok=True)

def create_validation_folder(folder="validation"):
    """Ensure the validation folder exists."""
    print(f"Ensuring validation folder exists: {folder}")
    os.makedirs(folder, exist_ok=True)


def run_python_script(script_path, venv_dir="venv"):
    """Run a Python script in the virtual environment."""
    python_executable = os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin", "python")
    print(f"Running Python script: {script_path}")
    subprocess.run([python_executable, script_path], check=True)


if __name__ == "__main__":
    venv_dir = "venv"
    requirements_file = "requirements.txt"

    try:
        create_venv(venv_dir)
        upgrade_pip(venv_dir)
        install_requirements(requirements_file, venv_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Setup failed!")
        sys.exit(1)
    create_dataset_folder()
    create_validation_folder()
    print("Setup completed!")
    print()
    run_python_script("verify_path.py")
