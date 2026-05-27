#!/usr/bin/env bash
# Phase 0 MVP runner.

set -euo pipefail

export N=${N:-3000}
export SEED=${SEED:-42}
export MODEL=${MODEL:-anthropic/claude-haiku-4-5}
export TTS_BACKEND=${TTS_BACKEND:-xtts_v2}

mkdir -p data/audio data/splits
echo "Config: N=$N SEED=$SEED MODEL=$MODEL TTS_BACKEND=$TTS_BACKEND"

echo "=== 1. Generate $N rows ==="
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

echo "=== 2. TTS ($TTS_BACKEND) ==="
PYTHONPATH=. python -u -c '
import os
from pathlib import Path
from src.tts_synth import synthesize_jsonl
synthesize_jsonl(
    Path("data/rows.jsonl"),
    Path("data/rows_audio.jsonl"),
    Path("data/audio"),
    seed=int(os.environ["SEED"]),
    backend_name=os.environ["TTS_BACKEND"],
)
'

echo "=== 3-8. Augment + ASR + curate + baselines + score + mine ==="
PYTHONPATH=. python -u scripts/finish_mvp.py --model "$MODEL"

echo ""
echo "MVP complete. Reports: data/report_*_mvp.json"
echo "Hardcases: data/hardcases_mvp.jsonl"
