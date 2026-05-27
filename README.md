# VoiceInvariantBench

**Speech-to-action conservation benchmark for voice agents.**

The question: when a person speaks a critical value — a date, time, amount, address, postcode, or code — does the agent preserve it all the way from audio to the final state of the world?

The metric: **SACR (Speech-to-Action Conservation Rate)** — the fraction of dialogues where every critical spoken invariant survives intact into the post-action state.

## Architecture

```
src/
  invariant_graph.py        # typed values + dependencies + per-turn lineage
  dialogue_plan.py          # turn structure, intent, action_on_invariant
  dialogue_render.py        # plan + graph → natural language (deterministic)
  policy_engine.py          # formal rules: when must the agent ask?
  provenance.py             # every row carries its full derivation
  contamination_checks.py   # canaries + overlap hashing for hidden eval
  hardcase_miner.py         # cluster failures → recipes → variants
  row_schema.py             # BenchmarkRow wrapping all the above
  scenarios.py              # domain × tool × invariant factories
  generate.py               # composes graph + plan + renderer → row
  tts_synth.py              # Coqui XTTS-v2
  augment.py                # phone codec, noise, reverb
  asr_audit.py              # faster-whisper recognition diagnostics
  agents.py                 # 4 baselines + sandbox tool execution
  verify.py                 # SACR, SCER, PCR, UCR, RIR, SDC, CAS
  curate.py                 # split assembly with provenance tagging
```

Tool schemas in `tools/`. CLI evaluator in `eval/`. Demo in `demo/`. Tests in `tests/`. Runner scripts in `scripts/`.

## Quick start

```bash
# Local (laptop, no GPU, no API keys)
pip install -r requirements.txt
PYTHONPATH=. python scripts/smoke.py
PYTHONPATH=. python -m pytest tests/ -v

# RunPod (A100/H100 80GB)
bash scripts/setup_runpod.sh
bash scripts/run_mvp.sh           # Phase 0 (~$30-70, 1-3 days)
bash scripts/run_phase1.sh        # Phase 1 (~$100-250, 2-4 weeks)
```

## Verified pipeline behavior (smoke test, 362 rows)

| Baseline | SACR | SCER | PolicyCompliance | UnsafeCommit | Repair | CAS |
|---|---|---|---|---|---|---|
| Oracle (policy-aware) | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| Broken (acts blindly, ~30% corruption) | 0.73 | 0.27 | 0.05 | 1.00 | 0.77 | 0.82 |

The policy engine cleanly separates "agent got the value right" (SACR) from "agent followed the right procedure" (PolicyCompliance) — they're independent failure modes.

The hardcase miner produced 55 variants across 11 patterns and explicitly flagged 5 patterns with no recipe registered (`status: no_recipe_registered`). Coverage gaps are visible, not hidden.

The contamination checker detects planted overlaps. Canary phrases unique to the hidden eval split let you catch leakage in submissions.

## Metrics

- **SACR** — Speech-to-Action Conservation Rate (primary)
- **ICR** — Invariant Conservation Rate (transcript-level, from ASR audit)
- **SCER** — Silent Critical Error Rate (acted, corrupted, sounded successful)
- **PCR** — Policy Compliance Rate
- **UCR** — Unsafe Commit Rate (acted when policy required asking)
- **RIR** — Repair Incorporation Rate (correction tasks only)
- **SDC** — Say-Do Consistency
- **CAS** — Counterfactual Action Sensitivity (paired rows)

## What's NOT included, deliberately

- **No invented `confidence_prior` field.** We record the risk factors we *injected* (`acoustic_risk`, `ambiguity_class`) — those are honest. Inventing prior probabilities for synthetic data and feeding them to the scorer would be a closed loop pretending to validate itself.
- **No multilingual rendering yet.** The planner/renderer split makes it tractable; doing it now would be premature.
- **No real-LLM baselines in the smoke test.** Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` and use `scripts/run_mvp.sh` to score actual models.
- **No Hugging Face push.** When ready, `huggingface-cli upload` the `data/splits/*.jsonl` files — but hold `hidden_eval.jsonl` privately.

## Design decisions worth knowing

- **Drop the 1M-row target.** Quality at 5-10k > volume at 1M. See `ROADMAP.md`.
- **Planner/renderer split** enables multilingual scaling without touching dialogue structure.
- **Policy engine is rule-based and formal**; UnsafeCommitRate is computed from it, not from string matching "confirm".
- **Hardcase miner surfaces its own coverage gaps** via `status: no_recipe_registered` — coverage debt is visible.
- **Provenance lives on every row**, not in sidecars, so it survives shuffle/filter/concatenate.

## Files

- `requirements.txt` — Python deps
- `scripts/smoke.py` — offline end-to-end test
- `scripts/setup_runpod.sh` — fresh-pod setup with model pre-download
- `scripts/run_mvp.sh` — Phase 0 (~3k rows)
- `scripts/run_phase1.sh` — sharded Phase 1 (~8k rows)
- `eval/score_predictions.py` — `voice-invariant-eval` CLI
- `demo/app.py` — Gradio Space showing silent-critical-error cases
- `tests/test_pipeline.py` — pytest suite
- `COSTS.md` — RunPod cost breakdown by phase
- `ROADMAP.md` — phased plan with the 1M-row target dropped
