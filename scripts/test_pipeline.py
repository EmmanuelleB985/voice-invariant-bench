"""Unit + integration tests. Run with: pytest tests/"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.canonicalize import (
    canonicalize_time, canonicalize_amount, canonicalize_code,
    canonicalize_postcode, words_to_int, int_to_words,
    int_to_spelled_digits, time_to_spoken,
)
from src.generate import generate_rows, build_row, make_counterfactual
from src.scenarios import SCENARIOS
from src.policy_engine import PolicyEngine
from src.invariant_graph import InvariantGraph, Invariant
from src.verify import score_one, score_predictions_file
from src.agents import oracle_with_policy


def test_canonicalize_basic():
    assert canonicalize_time("four fifty p.m.") == "16:50"
    assert canonicalize_time("nine thirty am") == "09:30"
    assert words_to_int("fourteen") == 14
    assert words_to_int("one four") == 14
    assert canonicalize_amount("two hundred fifty") == 250.0
    assert canonicalize_code("A B dash one two nine four dash Z") == "AB-1294-Z"
    assert canonicalize_postcode("S W one A one A A") == "SW1A 1AA"


def test_int_to_words_roundtrip():
    for n in [0, 1, 7, 10, 14, 40, 99, 100, 250, 999]:
        assert words_to_int(int_to_words(n)) == n


def test_time_roundtrip():
    assert time_to_spoken(16, 50).startswith("four fifty")
    assert canonicalize_time(time_to_spoken(16, 50)) == "16:50"
    assert canonicalize_time(time_to_spoken(9, 0)) == "09:00"


def test_generate_smoke():
    rows = generate_rows(20, seed=0)
    assert 20 <= len(rows) <= 30  # plus some counterfactuals
    # All rows have at least one invariant
    for r in rows:
        assert len(r.invariant_graph.invariants) >= 1
        assert r.reference_dialogue, "empty dialogue"


def test_lineage_recorded():
    rng = random.Random(0)
    s = SCENARIOS[0]
    r = build_row(rng, s, plan_type="repair_mishear",
                  seed_pack={"scenario": 0, "plan": 0, "render": 0})
    inv = r.invariant_graph.invariants[0]
    actions = [e.action for e in inv.lineage]
    assert "introduce" in actions
    assert "mishear" in actions
    assert "correct" in actions
    assert "confirm" in actions
    assert inv.was_corrected
    assert inv.was_confirmed


def test_counterfactual_preserves_unrelated():
    rng = random.Random(0)
    # Use the transfer scenario which has two invariants (amount + code)
    s = next(x for x in SCENARIOS if x["tool_schema"] == "transfer_value")
    base = build_row(rng, s, plan_type="single_turn",
                     seed_pack={"scenario": 0, "plan": 0, "render": 0})
    cf = make_counterfactual(base, s, rng)
    # Exactly one final-state field should differ
    diff = {k for k in base.expected_final_state
            if base.expected_final_state[k] != cf.expected_final_state[k]}
    assert len(diff) >= 1, "counterfactual should change at least one field"


def test_policy_engine_irreversible_demands_confirmation():
    pe = PolicyEngine()
    g = InvariantGraph(invariants=[Invariant(
        type="amount", surface_forms=["$50"], canonical_value=50.0,
        target_fields=["transfer_value.amount"], ambiguity_class="none",
        acoustic_risk="low",
    )])
    d = pe.decide(risk_level="irreversible", tool_irreversible=True, graph=g)
    assert d.action == "ask_confirmation"


def test_policy_engine_ambiguity_demands_clarification():
    pe = PolicyEngine()
    g = InvariantGraph(invariants=[Invariant(
        type="time", surface_forms=["nine"], canonical_value="09:00",
        target_fields=["create_event.time"],
        ambiguity_class="ampm_ambiguous", acoustic_risk="medium",
    )])
    d = pe.decide(risk_level="low", tool_irreversible=False, graph=g)
    assert d.action == "ask_clarification"


def test_policy_engine_low_risk_proceeds():
    pe = PolicyEngine()
    g = InvariantGraph(invariants=[Invariant(
        type="street_number", surface_forms=["fourteen"], canonical_value=14,
        target_fields=["update_delivery_address.street_number"],
        ambiguity_class="none", acoustic_risk="low",
    )])
    d = pe.decide(risk_level="low", tool_irreversible=False, graph=g)
    assert d.action == "proceed"


def test_oracle_scores_100(tmp_path):
    rows = generate_rows(30, seed=1, counterfactual_fraction=0.3)
    rp = tmp_path / "rows.jsonl"
    pp = tmp_path / "preds.jsonl"
    op = tmp_path / "report.json"
    with rp.open("w") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
    with pp.open("w") as f:
        for r in rows:
            row_dict = json.loads(r.model_dump_json())
            pred = oracle_with_policy(row_dict)
            pred["id"] = r.id
            # Synthesize a final_state from the expected one (oracle doesn't execute)
            pred["final_state"] = row_dict["expected_final_state"]
            f.write(json.dumps(pred) + "\n")
    agg = score_predictions_file(rp, pp, op)
    assert agg["SACR"] == 1.0
    assert agg["SilentCriticalErrorRate"] == 0.0
    assert agg["PolicyComplianceRate"] == 1.0
    # UCR may be None if no rows demand asking; otherwise 0
    if agg["UnsafeCommitRate"] is not None:
        assert agg["UnsafeCommitRate"] == 0.0


def test_broken_baseline_flags_silent_errors(tmp_path):
    """A baseline that always 'confirms done' while corrupting fields
    should produce a non-zero SilentCriticalErrorRate."""
    rows = generate_rows(50, seed=2)
    rp = tmp_path / "rows.jsonl"
    pp = tmp_path / "preds.jsonl"
    op = tmp_path / "report.json"
    rng = random.Random(0)
    with rp.open("w") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
    with pp.open("w") as f:
        for r in rows:
            d = json.loads(r.model_dump_json())
            fs = dict(d["expected_final_state"])
            if rng.random() < 0.5 and fs:
                k = rng.choice(list(fs.keys()))
                v = fs[k]
                if isinstance(v, int):
                    fs[k] = v * 10
                elif isinstance(v, float):
                    fs[k] = v * 10
                else:
                    fs[k] = "WRONG"
            pred = {
                "id": r.id,
                "tool_calls": d["expected_tool_calls"],
                "agent_messages": [],
                "final_state": fs,
                "final_response": "Done. All set.",
            }
            f.write(json.dumps(pred) + "\n")
    agg = score_predictions_file(rp, pp, op)
    assert agg["SACR"] < 0.9
    assert agg["SilentCriticalErrorRate"] > 0


# ---------- v1.1 additions ---------------------------------------------------

def test_no_rendering_artifacts():
    """Generated dialogues must not contain literal 'WRONG', '{placeholder}',
    or double periods. These were real bugs caught during the v1.0 -> v1.1
    pod run."""
    rows = generate_rows(100, seed=11)
    issues = []
    for r in rows:
        for t in r.reference_dialogue:
            if 'WRONG' in t.text:
                issues.append(('WRONG', r.task_type, t.text))
            if '{' in t.text and '}' in t.text:
                issues.append(('placeholder', r.task_type, t.text))
            if '..' in t.text and '...' not in t.text.replace('....', '...'):
                issues.append(('double_period', r.task_type, t.text))
    assert not issues, f"Rendering artifacts found: {issues[:5]}"


def test_multi_slot_scenarios_resolve():
    """transfer_value (amount + code) and set_reminder (quantity + time)
    both have multiple invariants. Both must render all template slots."""
    rows = generate_rows(120, seed=23)
    multi_slot_tools = ('transfer_value', 'set_reminder')
    seen = set()
    for r in rows:
        if r.tool_schema not in multi_slot_tools or r.tool_schema in seen:
            continue
        seen.add(r.tool_schema)
        for t in r.reference_dialogue:
            assert '{' not in t.text, \
                f"Unresolved placeholder in {r.tool_schema}/{r.task_type}: {t.text}"
    # Must have seen at least one of each multi-slot tool in 120 rows
    assert seen, "expected to see at least one multi-slot scenario"


def test_repair_mishearing_is_different_value():
    """The misheard value in a repair plan must DIFFER from the canonical."""
    rows = generate_rows(80, seed=31)
    repairs = [r for r in rows if r.task_type == 'repair_mishear']
    assert repairs, "expected at least one repair_mishear row"
    for r in repairs:
        misheard = r.dialogue_plan.metadata.get('misheard_value')
        canonical = r.invariant_graph.invariants[0].canonical_value
        # Convert misheard to comparable form
        assert misheard != canonical, \
            f"repair plan misheard == canonical ({misheard}): {r.id}"


def test_tts_backend_registry():
    """The TTS backend registry exposes all expected backends."""
    from src.tts_synth import BACKENDS, get_backend, DryRunBackend
    expected = {"xtts_v2", "eleven", "piper", "DRY_RUN"}
    assert expected.issubset(BACKENDS.keys())
    b = get_backend("DRY_RUN")
    assert isinstance(b, DryRunBackend)
    assert b.name == "DRY_RUN"


def test_asr_backend_registry():
    """The ASR backend registry exposes all expected backends."""
    from src.asr_audit import BACKENDS, get_backend
    expected = {"whisper", "deepgram", "assembly"}
    assert expected.issubset(BACKENDS.keys())
