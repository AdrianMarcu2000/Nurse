#!/usr/bin/env python3
"""One-shot script to build or rebuild the RAG index from data/protocols/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from nurse.rag.index import build_index

if __name__ == "__main__":
    force = "--force" in sys.argv
    print(f"Building RAG index (force={force})…")
    build_index(force=force)
    print("Done.")
