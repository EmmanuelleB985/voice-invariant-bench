"""Build the demo_data/ subset from a full MVP run.

Picks ~12 well-chosen examples covering silent errors, refusals, and
successes. Copies only those rows' audio (a few MB instead of GB).

Usage on the pod, after a successful MVP run:
    PYTHONPATH=. python scripts/build_demo_data.py

Produces demo_data/{examples.jsonl, audio/, preds_*.jsonl, report_*.json}.
scp the demo_data/ directory to your laptop, then it works as drop-in
input to demo/app.py — locally or on HF Spaces.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Source files — adjust if your MVP output naming differs
SRC_ROWS = Path("data/splits/benchmark_core.jsonl")
SRC_AUDIO_DIR = Path("data/audio")
SRC_PREDS = {
    "oracle":  Path("data/preds_oracle_v3.jsonl"),
    "asr_llm": Path("data/preds_asr_llm_v3.jsonl"),
    "clarify": Path("data/preds_clarify_v3.jsonl"),
}
SRC_REPORTS = {
    "oracle":  Path("data/report_oracle_v3.json"),
    "asr_llm": Path("data/report_asr_llm_v3.json"),
    "clarify": Path("data/report_clarify_v3.json"),
}

DST = Path("demo_data")
DST.mkdir(exist_ok=True)
DST_AUDIO = DST / "audio"
DST_AUDIO.mkdir(exist_ok=True)


def pick_curated_ids():
    """Return ~12 row IDs that tell the clearest story."""
    rows = {json.loads(l)["id"]: json.loads(l) for l in SRC_ROWS.open()}
    asr_preds = {json.loads(l)["id"]: json.loads(l) for l in SRC_PREDS["asr_llm"].open()}
    asr_report = json.loads(SRC_REPORTS["asr_llm"].read_text())
    per_row = {r["id"]: r for r in asr_report.get("per_row", [])}

    silent = []
    refusals_by_scenario = {}
    successes_by_scenario = {}

    for rid, row in rows.items():
        pred = asr_preds.get(rid, {})
        rec = per_row.get(rid, {})
        scenario = row.get("tool_schema", "?")

        if rec.get("silent_critical_error"):
            silent.append(rid)
        elif not pred.get("tool_calls") and scenario not in refusals_by_scenario:
            refusals_by_scenario[scenario] = rid
        elif rec.get("sacr_pass") and scenario not in successes_by_scenario:
            successes_by_scenario[scenario] = rid

    picked = []
    # All silent errors first (these are the headline)
    picked.extend(silent[:3])
    # One refusal per scenario (shows the per-scenario story)
    picked.extend(list(refusals_by_scenario.values())[:5])
    # One success per scenario (shows things work somewhere)
    picked.extend(list(successes_by_scenario.values())[:4])
    return picked, rows


def main():
    if not SRC_ROWS.exists():
        print(f"ERROR: {SRC_ROWS} not found. Run the MVP first.")
        sys.exit(1)

    picked, all_rows = pick_curated_ids()
    print(f"Picked {len(picked)} curated examples")

    # Write examples.jsonl
    with (DST / "examples.jsonl").open("w") as f:
        for rid in picked:
            row = all_rows[rid]
            f.write(json.dumps(row) + "\n")
    print(f"  -> {DST}/examples.jsonl")

    # Filter and copy predictions for just the picked IDs
    for name, src_path in SRC_PREDS.items():
        if not src_path.exists():
            print(f"  skipped {name}: {src_path} not found")
            continue
        dst = DST / f"preds_{name}.jsonl"
        with src_path.open() as fin, dst.open("w") as fout:
            for line in fin:
                p = json.loads(line)
                if p["id"] in set(picked):
                    fout.write(line)
        print(f"  -> {dst}")

    # Copy reports (full, not filtered — they're small)
    for name, src_path in SRC_REPORTS.items():
        if not src_path.exists():
            print(f"  skipped {name}: {src_path} not found")
            continue
        dst = DST / f"report_{name}.json"
        shutil.copy(src_path, dst)
        print(f"  -> {dst}")

    # Copy audio for picked rows only
    n_copied = 0
    for rid in picked:
        row = all_rows[rid]
        for turn in row.get("audio_dialogue", []):
            rel = turn.get("audio")
            if not rel:
                continue
            src = SRC_AUDIO_DIR / rel
            dst = DST_AUDIO / rel
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
                n_copied += 1
    print(f"  -> {DST_AUDIO} ({n_copied} audio files)")

    # Size summary
    total = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file())
    print(f"\nTotal demo_data size: {total / 1024:.1f} KB")
    print(f"\nTo deploy locally:")
    print(f"  python demo/app.py")
    print(f"\nTo deploy to HF Spaces:")
    print(f"  scp -r demo_data root@your-pod:/tmp/  # or just zip it")
    print(f"  Then upload to your HF Space alongside demo/app.py")


if __name__ == "__main__":
    main()