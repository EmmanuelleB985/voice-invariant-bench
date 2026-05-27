"""Curate generated rows into release configs.

Target sizes are guidance, not requirements: if generation produced less,
splits shrink proportionally. The mini and voice_agent_ci splits are
always populated if there's any input.
"""
from __future__ import annotations
import json
import random
from pathlib import Path


# Target sizes for a Phase 1 release. Scale up for v1/v2 as needed.
DEFAULT_TARGETS = {
    "mini": 200,
    "benchmark_core": 5_000,
    "benchmark_hard": 1_000,
    "multi_turn_repair": 2_000,
    "counterfactual_pairs": 1_000,    # row count (= 500 pairs × 2)
    "policy_confirmation": 1_000,
    "voice_agent_ci": 300,
    "hidden_eval": 500,
}


def curate(all_rows_path: Path, out_dir: Path,
           targets: dict[str, int] | None = None,
           seed: int = 42) -> dict:
    rng = random.Random(seed)
    targets = targets or DEFAULT_TARGETS
    rows = [json.loads(l) for l in all_rows_path.open()]
    rng.shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list] = {}
    for r in rows:
        by_task.setdefault(r["task_type"], []).append(r)

    splits = {k: [] for k in targets}

    # voice_agent_ci: high-severity rows (irreversible or risk != low)
    high_sev = [r for r in rows
                if r.get("tool_irreversible")
                or r.get("risk_level") in ("medium", "high", "irreversible")]
    splits["voice_agent_ci"] = high_sev[:targets["voice_agent_ci"]]

    # multi_turn_repair: union of repair_mishear, self_correct, conflicting_updates
    repair_rows = []
    for t in ("repair_mishear", "self_correct", "conflicting_updates"):
        repair_rows.extend(by_task.get(t, []))
    splits["multi_turn_repair"] = repair_rows[:targets["multi_turn_repair"]]

    # counterfactual_pairs: rows with a partner_id, plus their partners
    cf_rows = [r for r in rows if r.get("counterfactual_partner_id")]
    cf_ids = set()
    for r in cf_rows:
        cf_ids.add(r["id"])
        cf_ids.add(r["counterfactual_partner_id"])
    cf_full = [r for r in rows if r["id"] in cf_ids]
    splits["counterfactual_pairs"] = cf_full[:targets["counterfactual_pairs"]]

    # policy_confirmation: rows tagged for confirmation policy
    pc_rows = [r for r in rows
               if r.get("tool_irreversible")
               or r.get("risk_level") in ("high", "irreversible")]
    splits["policy_confirmation"] = pc_rows[:targets["policy_confirmation"]]

    # benchmark_core: balanced sample across task types
    splits["benchmark_core"] = rows[:targets["benchmark_core"]]

    # mini: tiny tour
    splits["mini"] = rows[:targets["mini"]]

    # hidden_eval: separate seed, no overlap with public splits
    public_ids = set()
    for k in ("mini", "benchmark_core", "multi_turn_repair", "voice_agent_ci"):
        public_ids.update(r["id"] for r in splits[k])
    hidden_pool = [r for r in rows if r["id"] not in public_ids]
    random.Random(seed + 999).shuffle(hidden_pool)
    splits["hidden_eval"] = hidden_pool[:targets["hidden_eval"]]

    # Tag provenance with split name
    manifest = {}
    for name, split_rows in splits.items():
        outp = out_dir / f"{name}.jsonl"
        with outp.open("w") as f:
            for r in split_rows:
                r.setdefault("provenance", {})["split"] = name
                f.write(json.dumps(r) + "\n")
        manifest[name] = len(split_rows)
        print(f"  {name}: {len(split_rows)} rows -> {outp}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
