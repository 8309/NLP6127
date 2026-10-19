"""Install dependencies only into this extracted delivery's virtual environment."""
from pathlib import Path
import subprocess
import sys
import venv


if __name__ == "__main__":
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Python 3.12 for this delivery.")
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "requirements-teacher.txt")], check=True)
    subprocess.run([str(python), str(root / "run_teacher.py"), "--check"], check=True, cwd=root)
    print("Setup complete. Launch RUN.cmd on Windows or run .venv/bin/python run_teacher.py.")
