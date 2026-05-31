#!/usr/bin/env bash
# Download all required model weights into the models/ directory.
# Run once before first launch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$ROOT/models"
mkdir -p "$MODELS_DIR"

echo "==> Downloading Whisper model (small.en via faster-whisper)…"
python3 -c "
from faster_whisper import WhisperModel
print('  Fetching small.en…')
WhisperModel('small.en', device='cpu', compute_type='int8')
print('  Done.')
"

echo "==> Downloading Qwen2.5-3B-Instruct Q4_K_M GGUF…"
MODEL_FILE="$MODELS_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    # Hugging Face direct download (no login required for this model)
    HF_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    echo "  Downloading from $HF_URL"
    curl -L --progress-bar -o "$MODEL_FILE" "$HF_URL"
    echo "  Saved to $MODEL_FILE"
else
    echo "  Already exists: $MODEL_FILE"
fi

echo "==> Downloading Kokoro TTS model…"
python3 -c "
from kokoro import KPipeline
print('  Fetching Kokoro…')
KPipeline(lang_code='a')
print('  Done.')
"

echo "==> Downloading BGE embedding model…"
python3 -c "
from sentence_transformers import SentenceTransformer
print('  Fetching BAAI/bge-small-en-v1.5…')
SentenceTransformer('BAAI/bge-small-en-v1.5')
print('  Done.')
"

echo ""
echo "All models downloaded successfully."
echo "Next step: python scripts/build_rag_index.py"
