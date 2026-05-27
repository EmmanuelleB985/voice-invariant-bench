"""VoiceInvariantBench — Gradio Demo

A single-page demo that tells the story from today's results:

  Tab 1: Listen — play a real audio dialogue, see oracle vs asr_llm outputs
         side by side, notice the silent corruption.
  Tab 2: Breakdown — per-scenario SACR for the three baselines, showing
         postcode/transfer collapse to 0.00 with Whisper.
  Tab 3: Policy axis — show clarify vs asr_llm UCR drop. The "buying
         confirmation costs you nothing on value preservation" finding.

Designed to work either locally (with full data/) or on HF Spaces (with a
sliced sample of data/). Falls back gracefully if files are missing.

Run locally:
    pip install gradio
    python demo/app.py

Deploy to HF Spaces:
    Copy this file to a Space, plus the demo_data/ directory with:
      - examples.jsonl   (10-20 hand-picked rows + their predictions)
      - audio/           (.wav files referenced in examples)
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import gradio as gr


# ---------- Data loading ----------------------------------------------------

def _find_first(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


ROWS_PATH = _find_first([
    "demo_data/examples.jsonl",
    "data/splits/benchmark_core.jsonl",
    "data/rows_audited.jsonl",
    "data/rows.jsonl",                       # smoke fallback
])
AUDIO_DIR = _find_first([
    "demo_data/audio",
    "data/audio",
])
PRED_FILES = {
    "oracle":  _find_first(["demo_data/preds_oracle.jsonl",
                              "data/preds_oracle_v3.jsonl",
                              "data/preds_oracle_mvp.jsonl",
                              "data/preds_oracle.jsonl"]),
    "asr_llm": _find_first(["demo_data/preds_asr_llm.jsonl",
                              "data/preds_asr_llm_v3.jsonl",
                              "data/preds_asr_llm_mvp.jsonl",
                              "data/preds_broken.jsonl"]),    # smoke fallback
    "clarify": _find_first(["demo_data/preds_clarify.jsonl",
                              "data/preds_clarify_v3.jsonl",
                              "data/preds_clarify_mvp.jsonl"]),
}
REPORT_FILES = {
    "oracle":  _find_first(["demo_data/report_oracle.json",
                              "data/report_oracle_v3.json",
                              "data/report_oracle_mvp.json",
                              "data/report_oracle.json"]),
    "asr_llm": _find_first(["demo_data/report_asr_llm.json",
                              "data/report_asr_llm_v3.json",
                              "data/report_asr_llm_mvp.json",
                              "data/report_broken.json"]),   # smoke fallback
    "clarify": _find_first(["demo_data/report_clarify.json",
                              "data/report_clarify_v3.json",
                              "data/report_clarify_mvp.json"]),
}


def _load_jsonl(p):
    if p is None:
        return {}
    out = {}
    with p.open() as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = row
    return out


ROWS = _load_jsonl(ROWS_PATH)
PREDS = {name: _load_jsonl(p) for name, p in PRED_FILES.items()}
REPORTS = {}
for name, p in REPORT_FILES.items():
    if p and p.exists():
        REPORTS[name] = json.loads(p.read_text())


# ---------- Curated examples for the Listen tab ----------------------------

def _pick_listen_examples():
    """Hand-pick rows: silent errors first, then refusals, then successes."""
    asr_pred = PREDS.get("asr_llm", {})
    asr_report = REPORTS.get("asr_llm", {})

    silent_errors = []
    refusals = []
    successes = []

    per_row = asr_report.get("per_row", []) if asr_report else []
    by_id = {r["id"]: r for r in per_row}

    for rid, row in ROWS.items():
        pred = asr_pred.get(rid, {})
        rec = by_id.get(rid, {})
        if rec.get("silent_critical_error"):
            silent_errors.append(rid)
        elif not pred.get("tool_calls"):
            refusals.append(rid)
        elif rec.get("sacr_pass"):
            successes.append(rid)

    picked = silent_errors[:3] + refusals[:5] + successes[:3]
    return picked or list(ROWS.keys())[:10]


LISTEN_IDS = _pick_listen_examples()


def listen_example(idx):
    if not LISTEN_IDS:
        return ("No examples available — run the MVP pipeline first.",
                None, "", "", "")
    idx = int(idx)
    rid = LISTEN_IDS[idx % len(LISTEN_IDS)]
    row = ROWS.get(rid, {})

    dialogue_md = "\n\n".join(
        f"**{t['speaker'].upper()}:** {t['text']}"
        for t in row.get("reference_dialogue", [])
    )

    audit = row.get("asr_audit") or {}
    transcript = audit.get("user_transcript", "(no ASR transcript)")

    audio_file = None
    if AUDIO_DIR:
        for turn in row.get("audio_dialogue", []):
            if turn["speaker"] == "user" and "audio" in turn:
                candidate = AUDIO_DIR / turn["audio"]
                if candidate.exists():
                    audio_file = str(candidate)
                    break

    comparison = []
    for name in ("oracle", "asr_llm", "clarify"):
        pred = PREDS.get(name, {}).get(rid, {})
        if not pred:
            continue
        tc = pred.get("tool_calls", [])
        final = pred.get("final_state", {})
        resp = (pred.get("final_response") or "")[:140]
        comparison.append(
            f"**{name.upper()}**\n\n"
            f"- Tool calls: `{json.dumps(tc) if tc else 'NONE (refused)'}`\n"
            f"- Final state: `{json.dumps(final)}`\n"
            f"- Response: *{resp}*\n"
        )
    comparison_md = "\n\n".join(comparison) if comparison else "*(no predictions)*"

    expected_md = (
        f"**Expected final state:** `{json.dumps(row.get('expected_final_state', {}))}`\n\n"
        f"**Expected tool call:** `{json.dumps(row.get('expected_tool_calls', []))}`"
    )

    rec = REPORTS.get("asr_llm", {}).get("per_row", [])
    row_rec = next((r for r in rec if r["id"] == rid), {})
    if row_rec.get("silent_critical_error"):
        kind = "**Silent critical error** — the agent acted, corrupted a value, and reported success."
    elif not PREDS.get("asr_llm", {}).get(rid, {}).get("tool_calls"):
        kind = "**Refusal** — the agent declined to call a tool. ASR likely destroyed the value."
    elif row_rec.get("sacr_pass"):
        kind = "**Success** — agent preserved the spoken value end-to-end."
    else:
        kind = "**Value corruption** — agent acted but with a wrong value."

    return (
        f"{kind}\n\n"
        f"**Scenario:** `{row.get('task_type', '?')}` / `{row.get('tool_schema', '?')}`",
        audio_file,
        f"### Dialogue (what the user spoke)\n\n{dialogue_md}",
        f"### ASR transcript (what the model received)\n\n```\n{transcript}\n```",
        f"### Baseline predictions\n\n{comparison_md}\n\n---\n\n{expected_md}",
    )


# ---------- Breakdown + summary tables -------------------------------------

def per_scenario_table_md():
    if not REPORTS:
        return "*No reports loaded.*"

    table = defaultdict(dict)
    for baseline, report in REPORTS.items():
        per_scenario = defaultdict(lambda: [0, 0])
        for rec in report.get("per_row", []):
            row = ROWS.get(rec["id"])
            if not row:
                continue
            key = row.get("tool_schema", "?")
            per_scenario[key][1] += 1
            if rec.get("sacr_pass"):
                per_scenario[key][0] += 1
        for k, (p, t) in per_scenario.items():
            table[k][baseline] = (p, t)

    if not table:
        return "*Could not compute per-scenario breakdown.*"

    lines = ["| Scenario | oracle | asr_llm | clarify |",
             "|---|---:|---:|---:|"]
    for scenario in sorted(table.keys()):
        row_cells = [f"`{scenario}`"]
        for baseline in ("oracle", "asr_llm", "clarify"):
            pt = table[scenario].get(baseline)
            if pt:
                p, t = pt
                row_cells.append(f"{p/t:.3f} ({p}/{t})")
            else:
                row_cells.append("—")
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def headline_summary_md():
    if not REPORTS:
        return "*No reports loaded — run the MVP pipeline first.*"

    rows = ["| Metric | oracle | asr_llm | clarify |",
            "|---|---:|---:|---:|"]
    metrics = ["SACR", "SilentCriticalErrorRate", "PolicyComplianceRate",
               "UnsafeCommitRate", "RepairIncorporationRate",
               "CounterfactualActionSensitivity"]
    for metric in metrics:
        cells = [metric]
        for baseline in ("oracle", "asr_llm", "clarify"):
            v = REPORTS.get(baseline, {}).get("summary", {}).get(metric)
            cells.append(f"{v:.3f}" if v is not None else "—")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------- Gradio interface -----------------------------------------------

INTRO_MD = """
# VoiceInvariantBench — Live Demo

**The question:** When a person speaks a critical value to a voice agent — a date, a time, an amount, an address, a postcode, a transfer code — does the agent preserve it all the way from audio to the final state of the world?

**The metric:** SACR (Speech-to-Action Conservation Rate) — the fraction of dialogues where every critical spoken value survives intact into the post-action state.

**Why a benchmark?** A single number like "voice agent X is 87% accurate" is a confound — it mixes TTS quality, ASR quality, and agent reasoning. This benchmark decomposes the stack: oracle measures the agent alone, asr_llm adds real ASR, and clarify adds a simple policy intervention. Three numbers, three independent failure modes.

→ Tab **Listen** plays real failures with audio.
→ Tab **Breakdown** shows which scenarios break and why.
→ Tab **Policy axis** shows that asking-before-acting is independent of value preservation.

Built on 146 dialogues across 6 scenarios, evaluated on Claude Haiku 4.5.
"""


with gr.Blocks(title="VoiceInvariantBench") as demo:
    gr.Markdown(INTRO_MD)

    with gr.Tabs():

        with gr.Tab("Listen"):
            gr.Markdown(
                "### What does a voice agent failure sound like?\n\n"
                "Each example below is a real dialogue from the benchmark. "
                "Listen to the user's audio, read the ASR transcript, then "
                "compare what the three baselines did. The 🚨 examples are "
                "*silent critical errors* — the agent acted, corrupted a "
                "value, and reported success."
            )

            idx_slider = gr.Slider(
                0, max(0, len(LISTEN_IDS) - 1), value=0, step=1,
                label=f"Example (1 of {len(LISTEN_IDS)})",
            )

            kind_md = gr.Markdown()
            audio = gr.Audio(label="User's spoken audio",
                              type="filepath", interactive=False)
            with gr.Row():
                with gr.Column():
                    dialogue_md = gr.Markdown()
                    transcript_md = gr.Markdown()
                with gr.Column():
                    comparison_md = gr.Markdown()

            idx_slider.change(
                listen_example, inputs=idx_slider,
                outputs=[kind_md, audio, dialogue_md, transcript_md,
                          comparison_md],
            )
            demo.load(listen_example, inputs=idx_slider,
                       outputs=[kind_md, audio, dialogue_md, transcript_md,
                                comparison_md])

        with gr.Tab("Breakdown"):
            gr.Markdown(
                "### Per-scenario SACR\n\n"
                "The headline finding. **Plain numeric values are preserved.** "
                "**Alphanumeric codes are destroyed by ASR.** This is the kind "
                "of stack-attributable failure single-number benchmarks miss.\n"
            )
            gr.Markdown(per_scenario_table_md())

            gr.Markdown(
                "\n\n### What the breakdown tells us\n\n"
                "- `create_event` and `update_delivery_address` are pure "
                "numeric values (times, house numbers). Whisper handles them "
                "fine; SACR stays high.\n"
                "- `update_postcode` and `transfer_value` involve letter-by-"
                "letter codes (e.g. *LO43 1VU*, *AB-1294-Z*). Whisper "
                "interprets these as English words (*Le 43, 1 view*; *Audi*), "
                "destroying the value before the agent ever sees it.\n"
                "- The agent layer is fine — oracle scores 0.88 across all "
                "scenarios. The collapse on codes is *entirely* a stack issue.\n\n"
                "**Procurement implication:** any voice agent doing "
                "transactions involving codes (account IDs, transfers, "
                "postcodes) needs ASR that handles letter-by-letter input. "
                "faster-whisper is not that ASR."
            )

        with gr.Tab("Policy axis"):
            gr.Markdown(
                "### Value correctness ≠ procedural correctness\n\n"
                "The benchmark separates two things most evals conflate:\n\n"
                "1. **Did the agent get the value right?** → SACR\n"
                "2. **Did the agent follow the right procedure (e.g. ask "
                "for confirmation on irreversible operations)?** → PCR / UCR\n\n"
                "A 4-line clarification heuristic (`clarification_policy_agent_v2`) "
                "demonstrates these are independent:\n"
            )

            gr.Markdown(headline_summary_md())

            gr.Markdown(
                "\n\n### Read the numbers\n\n"
                "- `clarify` SACR ≈ `asr_llm` SACR — adding clarification "
                "doesn't recover values that ASR has destroyed.\n"
                "- `clarify` PCR ≫ `asr_llm` PCR — but it dramatically "
                "changes procedural compliance.\n"
                "- `clarify` UCR ≪ `asr_llm` UCR — the agent stops "
                "committing without asking.\n\n"
                "**Finding:** a simple 4-line intervention closes most of "
                "the procedural gap without changing the recognition gap. "
                "Different fixes for different failure modes — exactly what "
                "decomposable metrics enable."
            )

        with gr.Tab("About"):
            gr.Markdown(
                "### How this works\n\n"
                "Every dialogue carries an **invariant graph** — a typed "
                "representation of each spoken value with per-turn lineage "
                "(who introduced it, who corrected it, who confirmed it). "
                "The renderer turns plans into natural language with no "
                "ambiguity about where each value comes from. The verifier "
                "scores against the canonical typed value, not against "
                "string-matching, so case/formatting doesn't matter.\n\n"
                "**Try it on your agent:**\n\n"
                "```bash\n"
                "pip install voice-invariant-bench\n"
                "voice-invariant-eval score \\\n"
                "    --dataset path/to/benchmark.jsonl \\\n"
                "    --predictions your_predictions.jsonl \\\n"
                "    --output report.json\n"
                "```\n\n"
                "**Scope of this preview:** 146 dialogues, 6 scenarios, one "
                "model (Claude Haiku 4.5), one TTS (XTTS-v2), one ASR "
                "(faster-whisper large-v3). Phase 1 release (5-10k rows, "
                "multi-backend stack matrix) is in progress.\n\n"
                "**License:** MIT for code, CC-BY-4.0 for the dataset.\n"
            )


if __name__ == "__main__":
    if not ROWS:
        print("WARNING: No data found.")
        print(f"  ROWS_PATH:   {ROWS_PATH}")
        print(f"  AUDIO_DIR:   {AUDIO_DIR}")
        print(f"  PRED_FILES:  {PRED_FILES}")
        print("Demo will start but show 'no data' messages.")
    else:
        print(f"Loaded {len(ROWS)} rows from {ROWS_PATH}")
        print(f"Listen tab cycles through {len(LISTEN_IDS)} curated examples")
        print(f"Predictions loaded: {[k for k, v in PREDS.items() if v]}")
        print(f"Reports loaded: {list(REPORTS.keys())}")

    demo.launch(server_name="0.0.0.0", server_port=7860,
                theme=gr.themes.Soft(),
                share=os.environ.get("GRADIO_SHARE") == "1")