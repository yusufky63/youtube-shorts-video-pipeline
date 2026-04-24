import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "ui" / "streamlit_app.py"


def main() -> int:
    command = [sys.executable, "-m", "streamlit", "run", str(APP_PATH)]
    return subprocess.call(command, cwd=ROOT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
