#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$ROOT"

if [ "$#" -lt 3 ]; then
  echo "Usage: zsh run_sortformer.sh <input_dir_or_manifest> <output_dir> <nemo_root> [mode] [python_bin] [manifest_flag]"
  echo "manifest_flag: dir (default) or manifest"
  exit 1
fi

INPUT_PATH="$1"
OUT_DIR="$2"
NEMO_ROOT="$3"
MODE="${4:-streaming}"
PYTHON_BIN="${5:-python}"
MANIFEST_FLAG="${6:-dir}"

mkdir -p "$OUT_DIR"

ARGS=(
  --output-dir "$OUT_DIR"
  --nemo-root "$NEMO_ROOT"
  --mode "$MODE"
)

if [ "$MANIFEST_FLAG" = "manifest" ]; then
  ARGS+=(--manifest "$INPUT_PATH")
else
  ARGS+=(--input-dir "$INPUT_PATH")
fi

"$PYTHON_BIN" "$PROJECT/run_sortformer_sada.py" "${ARGS[@]}"

echo
echo "Done."
echo "Outputs: $OUT_DIR"
