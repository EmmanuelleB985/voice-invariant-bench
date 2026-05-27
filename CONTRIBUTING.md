# Contributing to VoiceInvariantBench

Thanks for considering a contribution. This benchmark is designed to be extended adding scenarios, agents, and ASR/TTS backends is meant to be a one-file operation. The architecture exists so external contributors can plug things in without touching core logic.

## Quick start

```bash
git clone https://github.com/EmmanuelleB985/voice-invariant-bench
cd voice-invariant-bench
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/ -v       # all 17 should pass
PYTHONPATH=. python scripts/smoke.py          # end-to-end test, no GPU
```

If both pass, you're set up.

## What kinds of contributions are most welcome

In rough priority order:

1. **New TTS or ASR backends.** See `src/tts_synth.py` and `src/asr_audit.py`. Each backend is a class with one method.
2. **New hardcase recipes.** The miner produces `no_recipe_registered` patterns for failure modes nobody's written a generator for. Each recipe is ~10 lines.
3. **New scenarios.** Calendar, retail, transfer, reminder are seeded. Healthcare, IT helpdesk, customer service all fit the same pattern.
4. **Multilingual rendering.** The planner/renderer split is designed to support this — the plan is language-agnostic, only the renderer changes.
5. **Bug reports.** Especially around scoring edge cases or values that don't canonicalize cleanly.

## Adding a TTS backend

```python
# src/tts_synth.py
class MyTTSBackend(TTSBackend):
    name = "my_tts"
    speakers = ["voice_A", "voice_B"]

    def synth(self, text, out_path, speaker, speed=1.0):
        # Synthesize text to a .wav file at out_path.
        # Use self.speakers and the speaker argument for voice selection.
        ...

# Register in the BACKENDS dict:
BACKENDS["my_tts"] = MyTTSBackend
```

Run `pytest tests/test_pipeline.py::test_tts_backend_registry` to confirm. Run a small audio smoke with `bash scripts/run_mvp.sh N=10 TTS_BACKEND=my_tts` to confirm end-to-end.

## Adding an ASR backend

Same pattern in `src/asr_audit.py`:

```python
class MyASRBackend(ASRBackend):
    name = "my_asr"

    def transcribe(self, audio_path):
        # Return the transcribed text as a string.
        ...

BACKENDS["my_asr"] = MyASRBackend
```

Run a small audit on existing audio with `audit_jsonl(..., backend_name="my_asr")`.

## Adding a hardcase recipe

In `src/hardcase_miner.py`, add to `PATTERN_TO_RECIPES`:

```python
"postcode__spelling_dependent__single_turn": [
    lambda inv, scenario, rng: _spelling_dependent_variant(...)
]
```

Each lambda takes the invariant, the scenario, and an RNG; returns a new invariant or row. The miner clusters failures, looks up your recipe by pattern key, and generates variants.

## Adding a scenario

In `src/scenarios.py`, add to `SCENARIOS`:

```python
{
  "domain": "healthcare",
  "task_type": "schedule_visit",
  "tool_schema": "schedule_appointment",   # must exist in tools/
  "make_invariants": _make_visit_invariants,
  "make_args": _make_visit_args,
  "make_final": _make_visit_final,
  "initial_state": {...},
  "risk_level": "low",
}
```

The three `make_*` callables wire together. See existing scenarios for shape.

## Tests

Tests are in `tests/test_pipeline.py`. Add a test for any new feature; the suite must remain green for PRs to merge:

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

For changes that touch scoring, run `scripts/smoke.py` and confirm oracle SACR=1.0 (any change here is a scoring regression).

## Code style

- Type hints encouraged but not required.
- Keep modules focused (one concept per file).
- Comments explain *why*, code explains *what*.
- Avoid heavy dependencies. The pipeline runs on a laptop CPU for testing.

## Reporting issues

Good bug reports include:
- The exact command you ran
- What you expected
- What happened instead (full traceback if any)
- Your environment: OS, Python version, pip freeze of the venv

For scoring questions, attach the row and the prediction so reproduction is possible.
