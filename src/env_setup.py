import os
import sys
import subprocess
import pkg_resources
import platform

# List of required packages for the project
REQUIRED_PACKAGES = [
    "maples_dr",
    "monai",
    "torch",
    "numpy",
    "tqdm",
    "pyyaml",
    "scikit-learn",
    "tensorboard",
    "matplotlib",
]

# Directories to ensure exist for the project structure
REQUIRED_DIRS = [
    os.path.join(os.getcwd(), "data"),
    os.path.join(os.getcwd(), "models"),
    os.path.join(os.getcwd(), "outputs"),
    os.path.join(os.getcwd(), "configs"),
]

def install_packages():
    """
    Check for each required package and install it if missing.
    """
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    for pkg in REQUIRED_PACKAGES:
        if pkg.lower() not in installed_packages:
            print(f"[env_setup] Installing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        else:
            print(f"[env_setup] Package already installed: {pkg}")


def create_directories():
    """
    Create necessary project directories if they do not exist.
    """
    for directory in REQUIRED_DIRS:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"[env_setup] Directory ready: {directory}")
        except Exception as e:
            print(f"[env_setup] Failed to create directory {directory}: {e}")


def main():
    print(f"[env_setup] Starting environment setup on {platform.system()} {platform.release()}")
    install_packages()
    create_directories()
    print("[env_setup] Environment setup complete.")


if __name__ == "__main__":
    main()
