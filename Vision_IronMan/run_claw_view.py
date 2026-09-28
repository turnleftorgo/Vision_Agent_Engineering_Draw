from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> int:
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "web_backend.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8002",
                "--reload",
                "--no-access-log",
            ],
            cwd=PROJECT_DIR,
        ),
        subprocess.Popen(
            ["npm", "run", "dev:local"],
            cwd=PROJECT_DIR / "web_frontend",
        ),
    ]

    def stop(*_: object) -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    return next((process.returncode for process in processes if process.returncode), 0)


if __name__ == "__main__":
    raise SystemExit(main())
