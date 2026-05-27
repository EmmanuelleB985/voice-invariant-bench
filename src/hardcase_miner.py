"""Hardcase miner.

The pipeline that turns this from a static dataset into a *living* benchmark:

  baseline agents run on candidate rows
  -> failures are clustered by pattern
  -> each cluster spawns a "failure recipe"
  -> recipes are used to generate hard variants
  -> variants enter benchmark_hard

The mining loop is what separates a benchmark people care about from one
they don't.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from collections import defaultdict
from pydantic import BaseModel


class FailurePattern(BaseModel):
    pattern_id: str
    description: str
    triggering_features: dict
    n_examples: int
    severity: str
    example_ids: list[str] = []
    pass_rate_before_mining: float = 0.0


def _invs(r: dict) -> list[dict]:
    if "invariant_graph" in r:
        return r["invariant_graph"].get("invariants", [])
    return r.get("invariants", [])


def cluster_failures(rows: dict[str, dict],
                     report: dict) -> list[FailurePattern]:
    """Group per-row failures by (primary_invariant_type, ambiguity, task)."""
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for row_result in report.get("per_row", []):
        if row_result["sacr_pass"]:
            continue
        rid = row_result["id"]
        row = rows.get(rid, {})
        invs = _invs(row)
        primary_type = invs[0]["type"] if invs else "unknown"
        primary_ambig = (invs[0].get("ambiguity_class")
                         if invs else "none") or "none"
        task = row.get("task_type", "unknown")
        buckets[(primary_type, primary_ambig, task)].append(rid)

    patterns = []
    for (inv_type, ambig, task), ids in buckets.items():
        if len(ids) < 2:
            continue
        n_in_bucket = sum(
            1 for r in rows.values()
            if _invs(r) and _invs(r)[0]["type"] == inv_type
        )
        pass_rate = 1.0 - (len(ids) / max(n_in_bucket, 1))
        patterns.append(FailurePattern(
            pattern_id=f"{inv_type}__{ambig}__{task}",
            description=f"{task} failures on {inv_type} ({ambig})",
            triggering_features={
                "invariant_type": inv_type,
                "ambiguity_class": ambig,
                "task_type": task,
            },
            n_examples=len(ids),
            severity="high" if len(ids) > 5 else "medium",
            example_ids=ids[:20],
            pass_rate_before_mining=pass_rate,
        ))
    return sorted(patterns, key=lambda p: -p.n_examples)


class Recipe(BaseModel):
    name: str
    applies_to_pattern: str
    description: str


def _replace_in_row(row: dict, old, new):
    """Replace a canonical value across invariants, tool calls, and state.
    Marks the row as needing re-render."""
    invs = _invs(row)
    for inv in invs:
        if inv.get("canonical_value") == old:
            inv["canonical_value"] = new
    for tc in row.get("expected_tool_calls", []):
        for k, v in list(tc.get("arguments", {}).items()):
            if v == old:
                tc["arguments"][k] = new
    for k, v in list(row.get("expected_final_state", {}).items()):
        if v == old:
            row["expected_final_state"][k] = new
    prov = row.setdefault("provenance", {})
    prov.setdefault("notes", []).append("needs_rerender_after_hardcase")


def apply_recipe(row: dict, recipe: Recipe, rng: random.Random) -> dict:
    """Return a mutated copy of the row."""
    new_row = json.loads(json.dumps(row))
    new_row["id"] = f"{row['id']}_hard_{recipe.name}_{rng.randint(0,9999):04d}"
    new_row["task_type"] = "adversarial_hard"
    prov = new_row.setdefault("provenance", {})
    prov["parent_row_id"] = row["id"]
    prov["mined_from_failure"] = recipe.applies_to_pattern

    invs = _invs(new_row)

    if recipe.name == "decimal_shift_press":
        for inv in invs:
            if inv["type"] == "amount":
                v = inv["canonical_value"]
                new_val = float(v) / 10 if rng.random() < 0.5 else float(v) * 10
                _replace_in_row(new_row, v, new_val)
                inv["ambiguity_class"] = "decimal_shift"
                inv["acoustic_risk"] = "high"

    elif recipe.name == "phonetic_teens_tens":
        for inv in invs:
            if inv["type"] == "street_number":
                v = int(inv["canonical_value"])
                if 10 <= v <= 19:
                    new_val = v * 10 if v > 10 else 20
                elif 20 <= v <= 90 and v % 10 == 0:
                    new_val = v // 10 + 10
                else:
                    new_val = rng.choice([14, 40, 13, 30])
                _replace_in_row(new_row, v, new_val)
                inv["ambiguity_class"] = "phonetic_confusable"

    elif recipe.name == "tight_followup":
        new_row["reference_dialogue"].append({
            "speaker": "user",
            "text": "Quick — I need this done now.",
        })

    elif recipe.name == "buried_correction":
        rd = new_row["reference_dialogue"]
        for i, t in enumerate(rd):
            if "not " in t.get("text", "").lower():
                rd.insert(i + 1, {"speaker": "agent",
                                  "text": "Of course. Anything else before we proceed?"})
                rd.insert(i + 2, {"speaker": "user",
                                  "text": "No, that's everything."})
                break

    elif recipe.name == "spell_pressure":
        # Force spelling-dependent codes to be spelled aloud
        for inv in invs:
            if inv["type"] in ("code", "recipient_id", "postcode"):
                inv["ambiguity_class"] = "spelling_dependent"
                inv["acoustic_risk"] = "high"

    return new_row


PATTERN_TO_RECIPES: dict[str, list[Recipe]] = {
    "amount__none__single_turn": [
        Recipe(name="decimal_shift_press",
               applies_to_pattern="amount__none__single_turn",
               description="Force decimal-shift confusable"),
    ],
    "amount__decimal_shift__single_turn": [
        Recipe(name="decimal_shift_press",
               applies_to_pattern="amount__decimal_shift__single_turn",
               description="Stronger decimal shift"),
        Recipe(name="tight_followup",
               applies_to_pattern="amount__decimal_shift__single_turn",
               description="Pressure agent to act fast"),
    ],
    "street_number__none__single_turn": [
        Recipe(name="phonetic_teens_tens",
               applies_to_pattern="street_number__none__single_turn",
               description="14/40 style"),
    ],
    "street_number__phonetic_confusable__single_turn": [
        Recipe(name="phonetic_teens_tens",
               applies_to_pattern="street_number__phonetic_confusable__single_turn",
               description="Force teen/ten swap"),
    ],
    "code__spelling_dependent__single_turn": [
        Recipe(name="spell_pressure",
               applies_to_pattern="code__spelling_dependent__single_turn",
               description="Force spelling-dependent code"),
    ],
    "_multi_turn_repair": [
        Recipe(name="buried_correction",
               applies_to_pattern="_multi_turn_repair",
               description="Bury the correction under benign turns"),
    ],
}


def recipes_for_pattern(p: FailurePattern) -> list[Recipe]:
    direct = PATTERN_TO_RECIPES.get(p.pattern_id, [])
    task = p.triggering_features.get("task_type", "")
    if task in ("repair_mishear", "conflicting_updates", "self_correct"):
        direct = direct + PATTERN_TO_RECIPES.get("_multi_turn_repair", [])
    return direct


def mine_hardcases(rows_path: Path, report_path: Path,
                   out_path: Path, variants_per_pattern: int = 50,
                   seed: int = 0) -> dict:
    rng = random.Random(seed)
    rows = {json.loads(l)["id"]: json.loads(l) for l in rows_path.open()}
    report = json.loads(report_path.read_text())
    patterns = cluster_failures(rows, report)

    summary = {"patterns": [], "n_hardcases": 0}
    with out_path.open("w") as fout:
        for pat in patterns:
            recipes = recipes_for_pattern(pat)
            if not recipes:
                summary["patterns"].append({
                    "pattern": pat.pattern_id,
                    "recipes_used": 0,
                    "n_examples": pat.n_examples,
                    "status": "no_recipe_registered",
                })
                continue
            n_made = 0
            for ex_id in pat.example_ids:
                base = rows.get(ex_id)
                if not base:
                    continue
                for recipe in recipes:
                    if n_made >= variants_per_pattern:
                        break
                    mutated = apply_recipe(base, recipe, rng)
                    fout.write(json.dumps(mutated) + "\n")
                    n_made += 1
                if n_made >= variants_per_pattern:
                    break
            summary["patterns"].append({
                "pattern": pat.pattern_id,
                "recipes_used": len(recipes),
                "n_examples": pat.n_examples,
                "n_variants_made": n_made,
                "status": "ok",
            })
            summary["n_hardcases"] += n_made
    return summary
