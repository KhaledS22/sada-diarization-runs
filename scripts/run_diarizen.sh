#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$ROOT"

if [ "$#" -lt 2 ]; then
  echo "Usage: zsh run_diarizen.sh <input_dir_or_manifest> <output_dir> [python_bin] [manifest_flag]"
  echo "manifest_flag: dir (default) or manifest"
  exit 1
fi

INPUT_PATH="$1"
OUT_DIR="$2"
PYTHON_BIN="${3:-python}"
MANIFEST_FLAG="${4:-dir}"

mkdir -p "$OUT_DIR"

ARGS=(
  --output-dir "$OUT_DIR"
)

if [ "$MANIFEST_FLAG" = "manifest" ]; then
  ARGS+=(--manifest "$INPUT_PATH")
else
  ARGS+=(--input-dir "$INPUT_PATH")
fi

"$PYTHON_BIN" "$PROJECT/run_diarizen_sada.py" "${ARGS[@]}"

echo
echo "Done."
echo "Outputs: $OUT_DIR"
