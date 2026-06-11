# colab_setup.py

import os
import sys
import shutil
import subprocess

REPO_URL = "https://github.com/bbrisch/summer_school_prog_dm.git"
REPO_DIR = "summer_school_prog_dm"
REPO_PATH = f"/content/{REPO_DIR}"

def setup():
    # 1. install uv
    if shutil.which("uv") is None:
        os.system("curl -LsSf https://astral.sh/uv/install.sh | sh")

    os.environ["PATH"] += ":/root/.local/bin"

    # 2. clone repo
    if os.path.exists(REPO_PATH):
        os.chdir(REPO_PATH)
    else:
        os.chdir("/content")
        os.system(f"git clone {REPO_URL}")
        os.chdir(REPO_PATH)

    # 3. venv
    if not os.path.exists(".venv"):
        print("Creating venv...")
        os.system("uv venv .venv")

    # 4. sync deps
    os.system("uv sync --python .venv/bin/python")

    # 5. add site-packages
    venv_site_packages = subprocess.check_output(
        [
            ".venv/bin/python",
            "-c",
            "import site; print(site.getsitepackages()[0])"
        ],
        text=True,
    ).strip()

    sys.path.insert(0, venv_site_packages)

    print("✅ uv environment ready")