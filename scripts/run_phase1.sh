#!/usr/bin/env bash
# Phase 1: 5-10k row public benchmark with sharded generation.

set -euo pipefail

TARGET_TOTAL=${TARGET_TOTAL:-8000}
SHARDS=${SHARDS:-8}
SHARD_START=${SHARD_START:-0}
SHARD_END=${SHARD_END:-$SHARDS}
MODEL=${MODEL:-gpt-4o-mini}
PER_SHARD=$((TARGET_TOTAL / SHARDS))

mkdir -p data/audio data/splits data/shards

for s in $(seq $SHARD_START $((SHARD_END - 1))); do
    echo "=== Shard $s / $SHARDS ($PER_SHARD rows) ==="
    SHARD_DIR=data/shards/$s
    if [ -f "$SHARD_DIR/done" ]; then
        echo "  shard $s already done, skipping"
        continue
    fi
    mkdir -p $SHARD_DIR

    SEED=$((1000 + s))
    PYTHONPATH=. python - <<PY
from src.generate import generate_rows
from src.tts_synth import synthesize_jsonl
from src.augment import augment_jsonl
from src.asr_audit import audit_jsonl
from pathlib import Path

shard_dir = Path("$SHARD_DIR")
rows = generate_rows($PER_SHARD, seed=$SEED, counterfactual_fraction=0.2)
with (shard_dir / "rows.jsonl").open("w") as f:
    for r in rows:
        f.write(r.model_dump_json() + "\n")
print(f"  generated {len(rows)}")

synthesize_jsonl(shard_dir / "rows.jsonl",
                 shard_dir / "rows_audio.jsonl",
                 Path("data/audio"), seed=$SEED)
augment_jsonl(shard_dir / "rows_audio.jsonl",
              shard_dir / "rows_aug.jsonl",
              Path("data/audio"), seed=$SEED)
audit_jsonl(shard_dir / "rows_aug.jsonl", Path("data/audio"),
            shard_dir / "rows_done.jsonl")
PY
    touch $SHARD_DIR/done
done

echo "=== Merging shards ==="
cat data/shards/*/rows_done.jsonl > data/rows_all.jsonl
echo "  $(wc -l < data/rows_all.jsonl) rows total"

echo "=== Curating ==="
PYTHONPATH=. python -c "
from pathlib import Path
from src.curate import curate
curate(Path('data/rows_all.jsonl'), Path('data/splits'))
"

echo "=== Contamination check ==="
PYTHONPATH=. python -c "
from pathlib import Path
from src.contamination_checks import assert_no_overlap
assert_no_overlap(Path('data/splits/hidden_eval.jsonl'),
                  [Path('data/splits/benchmark_core.jsonl'),
                   Path('data/splits/benchmark_hard.jsonl'),
                   Path('data/splits/multi_turn_repair.jsonl'),
                   Path('data/splits/voice_agent_ci.jsonl')])
"

echo "=== Baselines on benchmark_core (model=$MODEL) ==="
PYTHONPATH=. python -c "
from pathlib import Path
from src.agents import run_baseline, text_oracle_agent, asr_to_llm_agent, clarification_policy_agent
for fn, name in [(text_oracle_agent, 'oracle'),
                 (asr_to_llm_agent, 'asr_llm'),
                 (clarification_policy_agent, 'clarify')]:
    run_baseline(Path('data/splits/benchmark_core.jsonl'),
                 Path(f'data/preds_{name}.jsonl'),
                 Path('tools'), fn, model='$MODEL')
"

echo "=== Score all baselines ==="
for name in oracle asr_llm clarify; do
    PYTHONPATH=. python eval/score_predictions.py score \
        --dataset data/splits/benchmark_core.jsonl \
        --predictions data/preds_${name}.jsonl \
        --output data/report_${name}.json
done

echo "=== Mine hardcases from the weakest baseline ==="
PYTHONPATH=. python -c "
from pathlib import Path
from src.hardcase_miner import mine_hardcases
s = mine_hardcases(Path('data/splits/benchmark_core.jsonl'),
                   Path('data/report_asr_llm.json'),
                   Path('data/hardcases_phase1.jsonl'),
                   variants_per_pattern=100)
print(f'  produced {s[\"n_hardcases\"]} hardcases across {len(s[\"patterns\"])} patterns')
for p in s['patterns']:
    print(f'    {p[\"pattern\"]}: n={p[\"n_examples\"]}, variants={p.get(\"n_variants_made\", 0)} ({p.get(\"status\", \"ok\")})')
"

echo ""
echo "Phase 1 complete."
echo "  Public splits: data/splits/"
echo "  Hidden split: data/splits/hidden_eval.jsonl (do NOT publish)"
echo "  Reports: data/report_*.json"
echo "  Mined hardcases: data/hardcases_phase1.jsonl"
