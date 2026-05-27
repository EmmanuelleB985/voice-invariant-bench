# Changelog

## v1.1.1 (May 2026, same-day fix)

External audit found two scoring bugs that were inflating SACR. Fixed.

### Critical fixes

- **`src/agents.py` `run_baseline`**: `final or row["expected_final_state"]` silently fell back to the *expected* final state when an agent returned no tool calls and the executor's output was a falsy empty dict. This meant empty-initial-state scenarios (calendar/create_event, reminder/set_reminder, etc) were scoring SACR=1.0 for silent agents. Fix: `final if final is not None else {}`. Regression test added.
- **`src/agents.py` `_build_user_message`**: `if "asr_audit" in row` returned True when the key existed with value None, then crashed accessing `row["asr_audit"]["user_transcript"]`. This broke `finish_mvp.py --skip_asr`. Fix: `row.get("asr_audit")` and check truthiness.

### Demo robustness

- `demo/app.py` now tries multiple paths in fallback order so it works regardless of which pipeline stage produced the data.

### Tests

- Added `test_empty_tool_calls_does_not_inflate_sacr` regression test. Total now 17 passing.

---


## v1.1.0 (May 2026)

First production-tested release. Eight bugs caught during a real Phase 0 MVP run on RunPod A100; all fixed. One major capability added (stack-decomposable benchmarking).

### Bug fixes

- **agents.py**: Stronger `SYSTEM_PROMPT` with worked example. Original prompt allowed models to return arbitrary JSON keys; oracle baseline scored 0.25 SACR because the schema wasn't enforced. v1.1 forces exact `{tool_calls, agent_messages, final_response}` shape.
- **dialogue_render.py**: Multi-invariant scenarios (transfer = amount + code, reminder = quantity + time) now render *all* template slots. Previously only the primary invariant resolved; `{sf_code}` and `{sf_time}` leaked into synthesized audio.
- **generate.py**: `_generate_mishearing()` helper replaces the literal `"WRONG"` string fallback in repair plans. Now produces type-aware plausible mishearings (decimal-shift for amounts, character swap for codes/postcodes, hour shift for times).
- **dialogue_render.py**: `_normalize_punctuation()` collapses `..` to `.` after template-canonical concatenation produces `p.m..`-style artifacts.
- **scripts/run_mvp.sh**: All environment variables are now `export`ed before child python processes spawn. The original `SEED=42 python -c "...seed=$SEED..."` form caused silent step failures when shell expansion produced `seed=)` for unset variables.
- **dialogue_render.py**: `self_correct` plan now converts slip values to spoken forms for non-int invariants (was rendering raw `10:15` instead of `quarter past ten a.m.`).

### New features

- **`src/tts_synth.py`**: Refactored to a backend interface. Built-in backends: `XTTSBackend`, `ElevenLabsBackend`, `PiperBackend`, `DryRunBackend`. Audio paths namespaced by backend name (`xtts_v2/...`, `eleven/...`) so one row can carry multiple backend's audio.
- **`src/asr_audit.py`**: Same pattern. Built-in backends: `FasterWhisperBackend`, `DeepgramBackend`, `AssemblyAIBackend`. ASR results stored as `asr_audits[{tts}__{asr}]` for per-cell scoring.
- **`src/stack_matrix.py`**: Runs (TTS × ASR × agent) matrix and emits per-cell SACR plus per-axis ranges (how much does ASR choice vs TTS choice move the score?). The headline industry-relevant capability.
- **`src/agents.py`**: Added `text_oracle_agent_v2`, `asr_to_llm_agent_v2`, `clarification_policy_agent_v2`. Use litellm's native `tools=` parameter instead of asking the model to JSON-serialize tool calls.
- **`scripts/finish_mvp.py`**: Resumes the MVP from `rows_audio.jsonl` onwards. Idempotent — re-running doesn't re-synthesize cached audio. Has `--skip_augment` and `--skip_asr` flags for partial runs.
- **`scripts/run_stack_matrix.sh`**: Shell wrapper for `stack_matrix.py` with sensible defaults.

### Dependency pins

`requirements.txt` now explicitly pins:
- `transformers>=4.40,<4.50` (Coqui TTS uses GPT2PreTrainedModel from this era)
- `tokenizers<0.21` (transformers 4.4x requires older tokenizers)
- `litellm>=1.40,<1.60` (1.60+ requires tokenizers>=0.21, conflicts with TTS)

Without these, fresh installs fail with `ModuleNotFoundError: No module named 'torch.distributed.tensor.device_mesh'` (transformers 4.50+ needs PyTorch 2.5; RunPod images ship PyTorch 2.4).

`setup_runpod.sh` uses `pip install --ignore-installed` to avoid the blinker/distutils uninstall error on Debian/Ubuntu base images.

### Tests

Added 5 tests, bringing total to 16/16 passing:
- `test_no_rendering_artifacts`: no WRONG, no `..`, no `{placeholder}` in 100 generated rows
- `test_multi_slot_scenarios_resolve`: transfer + reminder dialogues fill all slots
- `test_repair_mishearing_is_different_value`: misheard ≠ canonical
- `test_tts_backend_registry`: all expected TTS backends registered
- `test_asr_backend_registry`: all expected ASR backends registered

---

## v1.0.0 (May 2026)

Initial release. Architecture validated end-to-end with offline smoke tests:
- Oracle baseline: SACR 1.00 across all metrics
- Broken baseline: SACR 0.73 with 27% SilentCriticalErrorRate, 100% UnsafeCommitRate
- Hardcase miner: produces variants and surfaces `no_recipe_registered` patterns

11 tests passing. 39 KB bundle, ~400 lines of meaningful logic across 17 modules.
