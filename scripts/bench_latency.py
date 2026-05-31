#!/usr/bin/env python3
"""Run latency benchmark without a microphone."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

if __name__ == "__main__":
    import typer
    from nurse.main import app
    # Delegate to the built-in bench command
    sys.argv = ["nurse", "bench"] + sys.argv[1:]
    app()
