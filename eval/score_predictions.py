"""voice-invariant-eval CLI.

Usage:
  voice-invariant-eval score --dataset path/to/rows.jsonl \\
      --predictions preds.jsonl --output report.json

  voice-invariant-eval run --dataset path/to/rows.jsonl \\
      --agent_endpoint http://localhost:8000 --output report.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Make src importable when invoked from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.verify import score_predictions_file


def cmd_score(args):
    score_predictions_file(
        Path(args.dataset), Path(args.predictions), Path(args.output),
    )


def cmd_run(args):
    """POST each row to a hosted agent endpoint, then score the responses."""
    import requests
    preds_path = Path(args.output).with_suffix(".predictions.jsonl")
    with Path(args.dataset).open() as fin, preds_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            resp = requests.post(args.agent_endpoint, json={
                "id": row["id"],
                "dialogue": row["reference_dialogue"],
                "audio_dialogue": row.get("audio_dialogue", []),
                "tools": row["tool_schema"],
                "initial_state": row.get("initial_state", {}),
            }, timeout=60)
            pred = resp.json()
            pred["id"] = row["id"]
            fout.write(json.dumps(pred) + "\n")
    score_predictions_file(Path(args.dataset), preds_path, Path(args.output))


def main():
    p = argparse.ArgumentParser(prog="voice-invariant-eval")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("score", help="Score saved predictions against a dataset")
    s.add_argument("--dataset", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_score)

    r = sub.add_parser("run", help="Run a hosted agent endpoint and score")
    r.add_argument("--dataset", required=True)
    r.add_argument("--agent_endpoint", required=True)
    r.add_argument("--output", required=True)
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
