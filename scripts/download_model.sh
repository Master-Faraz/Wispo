#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-small.en}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="$ROOT/models"
TARGET_FILE="$TARGET_DIR/ggml-${MODEL}.bin"

mkdir -p "$TARGET_DIR"

if [[ -f "$TARGET_FILE" ]]; then
  echo "Model $TARGET_FILE already exists, skipping download."
  exit 0
fi

URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL}.bin"
echo "Downloading $MODEL from $URL..."
curl -L --fail --retry 3 --progress-bar -o "$TARGET_FILE" "$URL"
echo "Saved to $TARGET_FILE ($(du -h "$TARGET_FILE" | cut -f1))"
