"""Row generator: composes invariant graph + dialogue plan + renderer
into BenchmarkRows.

Also produces counterfactual partner rows on demand: given a base row,
mutate exactly one invariant and re-render. Counterfactual partners are
linked via counterfactual_partner_id and changed_invariant_id, which
the verifier uses to compute CounterfactualActionSensitivity.
"""
from __future__ import annotations
import random
from copy import deepcopy
from typing import Optional

from src.invariant_graph import InvariantGraph
from src.dialogue_plan import (
    DialoguePlan, plan_single_turn, plan_repair_mishear,
    plan_self_correct, plan_conflicting_updates,
    plan_unsafe_commit_temptation,
)
from src.dialogue_render import DialogueRenderer
from src.provenance import Provenance
from src.row_schema import BenchmarkRow, DialogueTurn, ToolCall
from src.scenarios import SCENARIOS


PLAN_TYPES = ["single_turn", "repair_mishear", "self_correct",
              "conflicting_updates", "unsafe_commit_temptation"]


def _generate_mishearing(inv, rng):
    """Produce a plausible misheard value for an invariant.

    Always returns something *different* from the canonical value. Falls
    back to invariant-type-specific perturbations when confusable_with
    is not provided.
    """
    cv = inv.canonical_value
    if inv.confusable_with is not None and inv.confusable_with != cv:
        return inv.confusable_with

    if isinstance(cv, int):
        return cv + rng.choice([-1, 1, 10, -10]) if cv > 10 else cv + rng.choice([1, 2, 10])

    if isinstance(cv, float):
        # Decimal-shift mishearing is common for amounts
        return cv * rng.choice([0.1, 10.0, 0.5, 2.0])

    if isinstance(cv, str) and cv:
        # Time: shift by an hour or flip am/pm
        if ":" in cv and len(cv) == 5:
            try:
                h, m = cv.split(":")
                h_new = (int(h) + rng.choice([1, -1, 12])) % 24
                return f"{h_new:02d}:{m}"
            except ValueError:
                pass
        # Code / postcode: swap one character to a different value
        for _ in range(8):
            idx = rng.randrange(len(cv))
            ch = cv[idx]
            if ch.isdigit():
                new_ch = str((int(ch) + rng.choice([1, 2, 3, 5])) % 10)
            elif ch.isalpha():
                new_ch = chr((ord(ch.upper()) - 65 + rng.choice([1, 2, 3])) % 26 + 65)
            else:
                continue
            candidate = cv[:idx] + new_ch + cv[idx+1:]
            if candidate != cv:
                return candidate
        return cv + "X"

    return cv


def _build_plan(plan_type: str, primary_inv, scenario: dict,
                rng: random.Random) -> tuple[DialoguePlan, Optional[object]]:
    """Returns (plan, optional_replacement_invariant).
    For conflicting_updates the second value supersedes the first — we
    return the replacement invariant so the caller can update the graph,
    expected_tool_calls, and expected_final_state accordingly. The original
    invariant's canonical_value is wired into the plan as the first turn's
    value_override so the dialogue stays self-consistent."""
    if plan_type == "repair_mishear":
        wrong = _generate_mishearing(primary_inv, rng)
        return plan_repair_mishear(primary_inv.id, wrong), None
    if plan_type == "self_correct":
        slip = _generate_mishearing(primary_inv, rng)
        return plan_self_correct(primary_inv.id, slip), None
    if plan_type == "conflicting_updates":
        original_value = primary_inv.canonical_value
        for _ in range(20):
            candidates = scenario["invariant_factory"](rng)
            replacement = next(
                (c for c in candidates if c.type == primary_inv.type), None,
            )
            if replacement and replacement.canonical_value != primary_inv.canonical_value:
                replacement.id = primary_inv.id
                plan = plan_conflicting_updates(primary_inv.id,
                                                 replacement.canonical_value)
                # Wire the first user turn's value_override to the ORIGINAL
                # value (the one being changed from). After replacement,
                # the graph holds the new value, so the renderer would
                # otherwise emit the new value for both turns.
                for turn in plan.turns:
                    if (turn.role == "user"
                            and turn.intent == "open_request"
                            and turn.value_override is None):
                        turn.value_override = original_value
                        break
                plan.metadata["original_value"] = original_value
                return plan, replacement
        return plan_single_turn(primary_inv.id), None
    if plan_type == "unsafe_commit_temptation":
        return plan_unsafe_commit_temptation(primary_inv.id), None
    return plan_single_turn(primary_inv.id), None


def build_row(rng: random.Random, scenario: dict,
              plan_type: str = "single_turn",
              seed_pack: dict | None = None) -> BenchmarkRow:
    sp = seed_pack or {"scenario": 0, "plan": 0, "render": 0}

    # 1. Invariant graph
    invs = scenario["invariant_factory"](rng)
    primary_inv = invs[0]

    # 2. Plan (may return a replacement for conflicting_updates)
    plan, replacement = _build_plan(plan_type, primary_inv, scenario, rng)
    if replacement is not None:
        invs[0] = replacement
        primary_inv = replacement
    graph = InvariantGraph(invariants=invs)

    # 3. Render (mutates graph: appends lineage)
    renderer = DialogueRenderer(scenario["tool_schema"], seed=sp["render"])
    rendered = renderer.render(plan, graph)
    dialogue = [DialogueTurn(speaker=r, text=t) for r, t in rendered]

    return BenchmarkRow(
        domain=scenario["domain"],
        task_type=plan.schema_id,
        risk_level=scenario["risk_level"],
        tool_irreversible=scenario["tool_irreversible"],
        invariant_graph=graph,
        dialogue_plan=plan,
        reference_dialogue=dialogue,
        tool_schema=scenario["tool_schema"],
        initial_state=scenario["initial_state"],
        expected_tool_calls=[
            ToolCall(tool=scenario["tool_schema"],
                     arguments=scenario["make_args"](invs)),
        ],
        expected_final_state=scenario["make_final"](invs),
        provenance=Provenance(
            scenario_seed=sp["scenario"],
            plan_seed=sp["plan"],
            render_seed=sp["render"],
            plan_schema_id=plan.schema_id,
        ),
    )


def make_counterfactual(row: BenchmarkRow, scenario: dict,
                        rng: random.Random) -> BenchmarkRow:
    """Mutate exactly one invariant and re-render against the same plan.

    The plan structure is preserved; only the invariant value changes.
    This is what makes CounterfactualActionSensitivity well-defined.
    """
    # Pick one invariant to mutate
    invs = list(row.invariant_graph.invariants)
    idx = rng.randrange(len(invs))
    target_inv = invs[idx]
    target_type = target_inv.type

    # Regenerate until we get a different canonical value
    for _ in range(20):
        candidates = scenario["invariant_factory"](rng)
        new_inv = next((c for c in candidates if c.type == target_type), None)
        if new_inv and new_inv.canonical_value != target_inv.canonical_value:
            # Use the same ID so plan refs still resolve
            new_inv.id = target_inv.id
            new_inv.dependencies = target_inv.dependencies
            invs[idx] = new_inv
            break
    else:
        # Couldn't generate a different one — return shallow clone unchanged
        return row.model_copy(deep=True)

    new_graph = InvariantGraph(invariants=invs)
    # Re-render
    renderer = DialogueRenderer(row.tool_schema, seed=row.provenance.render_seed + 1)
    rendered = renderer.render(row.dialogue_plan, new_graph)
    dialogue = [DialogueTurn(speaker=r, text=t) for r, t in rendered]

    # Rebuild expected_tool_calls and expected_final_state from invs
    args = scenario["make_args"](invs)
    final_state = scenario["make_final"](invs)

    cf = row.model_copy(deep=True, update={
        "id": row.id + "_cf",
        "invariant_graph": new_graph,
        "reference_dialogue": dialogue,
        "expected_tool_calls": [ToolCall(tool=row.tool_schema, arguments=args)],
        "expected_final_state": final_state,
        "counterfactual_partner_id": row.id,
        "counterfactual_changed_invariant_id": target_inv.id,
    })
    cf.provenance = cf.provenance.model_copy(update={
        "parent_row_id": row.id,
        "render_seed": row.provenance.render_seed + 1,
    })
    cf.provenance.add_note(f"counterfactual: changed {target_type} "
                           f"{target_inv.canonical_value} -> {invs[idx].canonical_value}")
    return cf


def generate_rows(n: int, seed: int = 42,
                  plan_distribution: dict[str, float] | None = None,
                  counterfactual_fraction: float = 0.2,
                  scenario_filter=None) -> list[BenchmarkRow]:
    """Generate a mixed batch of rows. counterfactual_fraction of base rows
    also produce a partner row, so the total is approximately
    n * (1 + counterfactual_fraction)."""
    rng = random.Random(seed)
    dist = plan_distribution or {
        "single_turn": 0.45, "repair_mishear": 0.3,
        "self_correct": 0.1, "conflicting_updates": 0.1,
        "unsafe_commit_temptation": 0.05,
    }
    plan_types = list(dist.keys())
    plan_weights = list(dist.values())

    eligible = SCENARIOS if scenario_filter is None else [
        s for s in SCENARIOS if scenario_filter(s)
    ]

    out: list[BenchmarkRow] = []
    for i in range(n):
        scenario = rng.choice(eligible)
        ptype = rng.choices(plan_types, weights=plan_weights)[0]
        sp = {"scenario": seed + i,
              "plan": seed + i * 3,
              "render": seed + i * 7}
        row = build_row(rng, scenario, ptype, sp)
        out.append(row)
        if rng.random() < counterfactual_fraction:
            out.append(make_counterfactual(row, scenario, rng))
    return out
