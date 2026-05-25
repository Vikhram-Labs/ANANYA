#!/usr/bin/env bash
# Usage: ./convert_to_gguf.sh <merged_model_dir> <output.gguf> [q4_k_m]
set -euo pipefail
MODEL_DIR="${1:?model dir}"
OUTFILE="${2:?outfile}"
QUANT="${3:-q4_k_m}"

if [ ! -d "llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp
fi

python llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$OUTFILE" --outtype f16
# Optional quantize pass:
# ./llama.cpp/llama-quantize "$OUTFILE" "${OUTFILE%.gguf}.${QUANT}.gguf" "$QUANT"
echo "GGUF written: $OUTFILE"
