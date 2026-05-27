"""Offline smoke test.

Runs the full pipeline in dry-run TTS mode with stub LLM predictions so it
works on a laptop with no GPU and no API keys.

    python scripts/smoke.py

End-to-end: generate -> dry-run TTS -> stub baselines (oracle + broken)
-> score -> mine hardcases -> contamination check.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate import generate_rows
from src.tts_synth import synthesize_jsonl
from src.verify import score_predictions_file
from src.hardcase_miner import mine_hardcases
from src.agents import oracle_with_policy
from src.contamination_checks import assert_no_overlap, find_overlap


def stub_oracle(rows_path: Path, out_path: Path):
    with rows_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            pred = oracle_with_policy(row)
            pred["id"] = row["id"]
            pred["final_state"] = row["expected_final_state"]
            fout.write(json.dumps(pred) + "\n")


def stub_broken(rows_path: Path, out_path: Path, seed: int = 0):
    """Acts without confirmation; corrupts ~30% of rows; reports success."""
    rng = random.Random(seed)
    with rows_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            fs = dict(row["expected_final_state"])
            if rng.random() < 0.3 and fs:
                k = rng.choice(list(fs.keys()))
                v = fs[k]
                if isinstance(v, int):
                    fs[k] = v * 10 if v < 100 else v + 1
                elif isinstance(v, float):
                    fs[k] = v * 10
                else:
                    fs[k] = "WRONG_INJECTED"
            pred = {
                "id": row["id"],
                "tool_calls": row["expected_tool_calls"],
                "agent_messages": [],
                "final_state": fs,
                "final_response": "Done. All set.",
            }
            fout.write(json.dumps(pred) + "\n")


def main():
    data = Path("data")
    data.mkdir(exist_ok=True)

    print("=== 1. Generate rows ===")
    rows = generate_rows(300, seed=42, counterfactual_fraction=0.2)
    rows_path = data / "rows.jsonl"
    with rows_path.open("w") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
    print(f"  generated {len(rows)} rows")

    from collections import Counter
    print(f"  plan types: {dict(Counter(r.task_type for r in rows))}")
    print(f"  domains:    {dict(Counter(r.domain for r in rows))}")

    # Sanity check: no rendering artifacts
    artifacts = sum(
        1 for r in rows for t in r.reference_dialogue
        if "WRONG" in t.text or "{" in t.text or ".." in t.text
    )
    if artifacts:
        print(f"  WARNING: {artifacts} rows have rendering artifacts")
    else:
        print("  rendering clean (no WRONG, {placeholder}, or `..`)")

    print("\n=== 2. TTS (dry-run) ===")
    synthesize_jsonl(rows_path, data / "rows_audio.jsonl",
                     data / "audio", dry_run=True)

    print("\n=== 3. Stub baselines ===")
    stub_oracle(rows_path, data / "preds_oracle.jsonl")
    stub_broken(rows_path, data / "preds_broken.jsonl")

    print("\n=== 4. Score: oracle ===")
    score_predictions_file(rows_path, data / "preds_oracle.jsonl",
                            data / "report_oracle.json")

    print("\n=== 5. Score: broken ===")
    score_predictions_file(rows_path, data / "preds_broken.jsonl",
                            data / "report_broken.json")

    print("\n=== 6. Mine hardcases from broken baseline ===")
    summary = mine_hardcases(
        rows_path, data / "report_broken.json",
        data / "hardcases.jsonl", variants_per_pattern=10,
    )
    print(f"  produced {summary['n_hardcases']} hardcases "
          f"across {len(summary['patterns'])} patterns")
    for p in summary["patterns"]:
        st = p.get("status", "ok")
        print(f"  {p['pattern']}: n={p['n_examples']}, "
              f"variants={p.get('n_variants_made', 0)} ({st})")

    print("\n=== 7. Contamination check ===")
    public = data / "public_for_check.jsonl"
    hidden = data / "hidden_for_check.jsonl"
    with public.open("w") as f:
        for r in rows[:50]:
            f.write(r.model_dump_json() + "\n")
    with hidden.open("w") as f:
        for r in rows[100:150]:
            f.write(r.model_dump_json() + "\n")
    assert_no_overlap(hidden, [public])

    bad = data / "bad_hidden.jsonl"
    with bad.open("w") as f:
        for r in rows[:5]:
            f.write(r.model_dump_json() + "\n")
    overlaps = find_overlap(bad, [public])
    assert overlaps, "contamination detector failed!"
    print(f"  flagged {len(overlaps)} planted overlaps as expected")

    print("\n=== Done. All artifacts in data/ ===")


if __name__ == "__main__":
    main()
