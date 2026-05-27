#!/usr/bin/env bash
# Run a (TTS × ASR × agent) matrix on a small benchmark subset.
#
# Required env vars depending on backends selected:
#   ANTHROPIC_API_KEY        for Claude models via litellm
#   ELEVENLABS_API_KEY       for ElevenLabs TTS
#   DEEPGRAM_API_KEY         for Deepgram ASR
#   ASSEMBLYAI_API_KEY       for AssemblyAI ASR
set -euo pipefail

export N=${N:-200}
export SEED=${SEED:-42}
export MODEL=${MODEL:-anthropic/claude-haiku-4-5}
TTS_BACKENDS=${TTS_BACKENDS:-xtts_v2 eleven}
ASR_BACKENDS=${ASR_BACKENDS:-whisper deepgram}
OUT_DIR=${OUT_DIR:-data/matrix}

echo "Config:"
echo "  N=$N SEED=$SEED MODEL=$MODEL"
echo "  TTS_BACKENDS=$TTS_BACKENDS"
echo "  ASR_BACKENDS=$ASR_BACKENDS"
echo "  OUT_DIR=$OUT_DIR"

mkdir -p data
if [ ! -f data/rows.jsonl ] || [ "$(wc -l < data/rows.jsonl)" -lt "$N" ]; then
    echo "=== Generate $N rows ==="
    PYTHONPATH=. python -u -c '
import os
from pathlib import Path
from src.generate import generate_rows
N = int(os.environ["N"])
SEED = int(os.environ["SEED"])
rows = generate_rows(N, seed=SEED, counterfactual_fraction=0.2)
with Path("data/rows.jsonl").open("w") as f:
    for r in rows:
        f.write(r.model_dump_json() + "\n")
print(f"  wrote {len(rows)} rows")
'
fi

echo "=== Run stack matrix ==="
PYTHONPATH=. python -u -m src.stack_matrix \
    --rows data/rows.jsonl \
    --out_dir "$OUT_DIR" \
    --tts $TTS_BACKENDS \
    --asr $ASR_BACKENDS \
    --model "$MODEL"

echo ""
echo "Matrix complete. Reports: $OUT_DIR/reports/, summary: $OUT_DIR/summary.json"
