"""Stack matrix runner.

The headline Phase 1 capability: take one set of dialogues, render with M
TTS backends, transcribe with N ASR backends, then for each agent
baseline emit M×N reports. This is what makes a benchmark *useful for
procurement decisions* — letting people compare voice agent components
in like-for-like settings rather than trusting the vendor's marketing.

Output:
  data/matrix/
    rows_audited.jsonl                    (one row, all backends)
    reports/<agent>__<tts>__<asr>.json    (M*N*A reports)
    summary.json                          (the M*N*A grid + deltas)
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

from src.tts_synth import synthesize_jsonl_multi
from src.asr_audit import audit_jsonl
from src.agents import run_baseline
from src.verify import score_predictions_file


def run_stack_matrix(
    rows_path: Path,
    out_dir: Path,
    tts_backends: list[str],
    asr_backends: list[str],
    agents: dict[str, Callable],
    tools_dir: Path,
    model: str = "anthropic/claude-haiku-4-5",
    seed: int = 0,
) -> dict:
    """Run a (TTS × ASR × agent) matrix.

    Returns a summary dict with per-cell metrics and cross-cell deltas.

    Cost note: for M TTS × N ASR × A agents × R rows you pay:
      - M TTS renderings of all R rows
      - M × N ASR transcriptions of all R rows × ~5 turns
      - M × N × A agent evaluations of R rows
    Scale carefully. The default (M=2, N=2, A=3, R=300) is ~$3-5.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    preds_dir = out_dir / "preds"
    preds_dir.mkdir(exist_ok=True)

    # Phase 1: synthesize all TTS backends in one pass over the rows file
    print(f"\n=== Stack matrix: TTS pass over {tts_backends} ===")
    audio_jsonl = out_dir / "rows_audio.jsonl"
    synthesize_jsonl_multi(rows_path, audio_jsonl, audio_dir,
                            backend_names=tts_backends, seed=seed)

    # Phase 2: ASR audit for each (TTS, ASR) cell
    # Each pass reads the previous pass's output so audits accumulate.
    current = audio_jsonl
    for tts in tts_backends:
        for asr in asr_backends:
            print(f"\n=== ASR pass: tts={tts}, asr={asr} ===")
            next_path = out_dir / f"rows_audited_{tts}__{asr}.jsonl"
            audit_jsonl(current, audio_dir, next_path,
                        backend_name=asr, tts_backend=tts)
            current = next_path
    final_rows = out_dir / "rows_audited.jsonl"
    current.rename(final_rows)

    # Phase 3: For each cell, write a "selected-cell" version of the rows
    # so the agent reads the right ASR transcript, then run agents + score.
    summary: dict[str, dict] = {"cells": {}, "agents": list(agents.keys()),
                                "tts_backends": tts_backends,
                                "asr_backends": asr_backends}

    for tts in tts_backends:
        for asr in asr_backends:
            cell = f"{tts}__{asr}"
            print(f"\n=== Cell {cell} ===")
            # Project the cell-specific audit into the legacy asr_audit field
            # so agents see the right transcript when use_transcript=True.
            cell_rows_path = out_dir / f"rows_for_{cell}.jsonl"
            _project_cell(final_rows, cell_rows_path, cell_key=cell)
            for agent_name, agent_fn in agents.items():
                preds_path = preds_dir / f"{agent_name}__{cell}.jsonl"
                report_path = reports_dir / f"{agent_name}__{cell}.json"
                print(f"  agent={agent_name}")
                run_baseline(cell_rows_path, preds_path, tools_dir,
                             agent_fn, model=model)
                agg = score_predictions_file(cell_rows_path, preds_path,
                                              report_path)
                summary["cells"][f"{agent_name}__{cell}"] = agg

    # Compute summary deltas
    summary["deltas"] = _compute_deltas(summary["cells"], tts_backends,
                                         asr_backends, list(agents.keys()))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out_dir / 'summary.json'}")
    return summary


def _project_cell(in_path: Path, out_path: Path, cell_key: str):
    """Write a rows file where row['asr_audit'] points to the specific
    cell_key's transcript (rather than whichever was set last)."""
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            audits = row.get("asr_audits", {})
            if cell_key in audits:
                row["asr_audit"] = audits[cell_key]
            fout.write(json.dumps(row) + "\n")


def _compute_deltas(cells: dict, tts_list: list[str], asr_list: list[str],
                     agents: list[str]) -> dict:
    """Compute interpretable deltas across the cells dict.

    For each agent: max - min SACR across cells (variance attributable to
    stack choice). Also per-axis: holding TTS fixed, what's the ASR-induced
    variance? Holding ASR fixed, what's the TTS-induced variance?
    """
    deltas = {}
    for agent in agents:
        agent_cells = {k: v for k, v in cells.items()
                       if k.startswith(f"{agent}__")}
        sacrs = {k: v.get("SACR") for k, v in agent_cells.items()
                 if v.get("SACR") is not None}
        if not sacrs:
            continue
        sacr_values = list(sacrs.values())
        deltas[agent] = {
            "SACR_range": max(sacr_values) - min(sacr_values),
            "SACR_min_cell": min(sacrs, key=sacrs.get),
            "SACR_max_cell": max(sacrs, key=sacrs.get),
            "SACR_mean": sum(sacr_values) / len(sacr_values),
        }
        # Per-axis variance: hold TTS fixed, vary ASR
        per_tts_ranges = []
        for tts in tts_list:
            tts_sacrs = [sacrs[f"{agent}__{tts}__{asr}"]
                         for asr in asr_list
                         if f"{agent}__{tts}__{asr}" in sacrs]
            if len(tts_sacrs) > 1:
                per_tts_ranges.append(max(tts_sacrs) - min(tts_sacrs))
        if per_tts_ranges:
            deltas[agent]["SACR_asr_axis_avg_range"] = \
                sum(per_tts_ranges) / len(per_tts_ranges)
        # Hold ASR fixed, vary TTS
        per_asr_ranges = []
        for asr in asr_list:
            asr_sacrs = [sacrs[f"{agent}__{tts}__{asr}"]
                         for tts in tts_list
                         if f"{agent}__{tts}__{asr}" in sacrs]
            if len(asr_sacrs) > 1:
                per_asr_ranges.append(max(asr_sacrs) - min(asr_sacrs))
        if per_asr_ranges:
            deltas[agent]["SACR_tts_axis_avg_range"] = \
                sum(per_asr_ranges) / len(per_asr_ranges)
    return deltas


def print_matrix(summary: dict):
    """Pretty-print a (TTS × ASR) SACR matrix per agent."""
    cells = summary["cells"]
    for agent in summary["agents"]:
        print(f"\n  {agent}:")
        header = "    " + " " * 14 + " ".join(
            f"{a:>14s}" for a in summary["asr_backends"]
        )
        print(header)
        for tts in summary["tts_backends"]:
            row = [f"    {tts:>14s}"]
            for asr in summary["asr_backends"]:
                key = f"{agent}__{tts}__{asr}"
                v = cells.get(key, {}).get("SACR")
                row.append(f"{v:14.3f}" if v is not None else f"{'--':>14s}")
            print(" ".join(row))
        d = summary.get("deltas", {}).get(agent, {})
        if d:
            print(f"      SACR range (overall): {d.get('SACR_range', 0):.3f}")
            print(f"      ASR-axis avg range:   "
                  f"{d.get('SACR_asr_axis_avg_range', 0):.3f}")
            print(f"      TTS-axis avg range:   "
                  f"{d.get('SACR_tts_axis_avg_range', 0):.3f}")


if __name__ == "__main__":
    import sys, argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.agents import (
        text_oracle_agent_v2, asr_to_llm_agent_v2,
        clarification_policy_agent_v2,
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True,
                     help="Path to rows.jsonl input")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tts", nargs="+", default=["xtts_v2"])
    ap.add_argument("--asr", nargs="+", default=["whisper"])
    ap.add_argument("--model", default="anthropic/claude-haiku-4-5")
    args = ap.parse_args()

    summary = run_stack_matrix(
        Path(args.rows), Path(args.out_dir),
        tts_backends=args.tts, asr_backends=args.asr,
        agents={
            "oracle": text_oracle_agent_v2,
            "asr_llm": asr_to_llm_agent_v2,
            "clarify": clarification_policy_agent_v2,
        },
        tools_dir=Path("tools"),
        model=args.model,
    )
    print_matrix(summary)
