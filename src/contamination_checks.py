"""Contamination & leakage detection for the hidden eval split.

Two protections:
  1. Canaries: rows with a fixed, unusual phrase that must never appear
     in any public split. If a submission's predictions reveal recognition
     of the canary, the submitter trained on the hidden split.
  2. Overlap detection: hash-based fingerprint matching of (dialogue text,
     expected canonical values) between hidden and public splits.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Iterable


CANARY_PHRASE = "the canary phrase is CANARY-7F3A-VIB"
CANARY_RESPONSE_TOKEN = "CANARY-RESPONSE-9E2D-VIB"


def make_canary_row(template_row: dict) -> dict:
    row = json.loads(json.dumps(template_row))
    for turn in row.get("reference_dialogue", []):
        if turn["speaker"] == "user":
            turn["text"] = turn["text"] + f" (Note: {CANARY_PHRASE})"
            break
    row.setdefault("provenance", {})["notes"] = (
        row.get("provenance", {}).get("notes", []) + ["is_canary"]
    )
    return row


def predicted_response_leaked_canary(prediction: dict) -> bool:
    text = (prediction.get("final_response") or "") + " " + json.dumps(
        prediction.get("tool_calls", []),
    )
    return CANARY_RESPONSE_TOKEN in text or "CANARY-7F3A" in text


def _fingerprint(row: dict) -> str:
    parts = []
    for turn in row.get("reference_dialogue", []):
        parts.append(turn["text"].lower().strip())
    invs = (row.get("invariant_graph", {}).get("invariants", [])
            or row.get("invariants", []))
    for inv in invs:
        parts.append(str(inv.get("canonical_value", inv.get("canonical"))))
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:16]


def find_overlap(hidden_path: Path,
                 public_paths: Iterable[Path]) -> list[tuple[str, str]]:
    public_fps = {}
    for p in public_paths:
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                row = json.loads(line)
                public_fps[_fingerprint(row)] = row["id"]
    overlaps = []
    with hidden_path.open() as f:
        for line in f:
            row = json.loads(line)
            fp = _fingerprint(row)
            if fp in public_fps:
                overlaps.append((row["id"], public_fps[fp]))
    return overlaps


def assert_no_overlap(hidden_path: Path,
                      public_paths: Iterable[Path]) -> None:
    overlaps = find_overlap(hidden_path, public_paths)
    if overlaps:
        raise RuntimeError(
            f"Contamination: {len(overlaps)} hidden rows overlap "
            f"public splits. First: {overlaps[:3]}"
        )
    n = sum(1 for _ in hidden_path.open())
    print(f"OK: no overlap between hidden ({n} rows) and public splits")
