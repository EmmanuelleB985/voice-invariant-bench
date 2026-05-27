"""Resume MVP from rows_audio.jsonl onwards."""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment import augment_jsonl
from src.asr_audit import audit_jsonl
from src.curate import curate
from src.agents import (
    run_baseline,
    text_oracle_agent, asr_to_llm_agent, clarification_policy_agent,
)
from src.verify import score_predictions_file
from src.hardcase_miner import mine_hardcases

# Try to import v2 (tool-calling) agents; fall back if not present
try:
    from src.agents import (
        text_oracle_agent_v2, asr_to_llm_agent_v2,
        clarification_policy_agent_v2,
    )
    HAS_V2 = True
except ImportError:
    HAS_V2 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", default=os.environ.get("MODEL", "anthropic/claude-haiku-4-5"))
    ap.add_argument("--skip_augment", action="store_true")
    ap.add_argument("--skip_asr", action="store_true")
    ap.add_argument("--use_tool_calling", action="store_true", default=True)
    args = ap.parse_args()

    data = Path(args.data_dir)
    print(f"Using model: {args.model}", flush=True)
    print(f"Tool-calling agents available: {HAS_V2}", flush=True)

    if not args.skip_augment:
        print("\n=== Acoustic augmentation ===", flush=True)
        augment_jsonl(data / "rows_audio.jsonl",
                      data / "rows_augmented.jsonl",
                      data / "audio", seed=42)
        current = data / "rows_augmented.jsonl"
    else:
        current = data / "rows_audio.jsonl"
        if (data / "rows_augmented.jsonl").exists():
            current = data / "rows_augmented.jsonl"

    if not args.skip_asr:
        print("\n=== ASR audit ===", flush=True)
        audit_jsonl(current, data / "audio", data / "rows_audited.jsonl")
        current = data / "rows_audited.jsonl"
    elif (data / "rows_audited.jsonl").exists():
        current = data / "rows_audited.jsonl"

    print("\n=== Curate ===", flush=True)
    n_rows = sum(1 for _ in current.open())
    targets = {k: min(v, n_rows) for k, v in {
        "mini": 100, "benchmark_core": 200, "benchmark_hard": 50,
        "multi_turn_repair": 80, "counterfactual_pairs": 40,
        "policy_confirmation": 40, "voice_agent_ci": 50,
        "hidden_eval": 50,
    }.items()}
    curate(current, data / "splits", targets=targets)

    bc = data / "splits" / "benchmark_core.jsonl"
    print(f"benchmark_core: {sum(1 for _ in bc.open())} rows", flush=True)

    if args.use_tool_calling and HAS_V2:
        agents = [("oracle", text_oracle_agent_v2),
                  ("asr_llm", asr_to_llm_agent_v2),
                  ("clarify", clarification_policy_agent_v2)]
        print("Using v2 tool-calling agents", flush=True)
    else:
        agents = [("oracle", text_oracle_agent),
                  ("asr_llm", asr_to_llm_agent),
                  ("clarify", clarification_policy_agent)]
        print("Using v1 prompt-based agents", flush=True)

    for name, fn in agents:
        print(f"\n=== Baseline: {name} ===", flush=True)
        run_baseline(bc, data / f"preds_{name}_v3.jsonl",
                     Path("tools"), fn, model=args.model)
        print(f"\n=== Score: {name} ===", flush=True)
        score_predictions_file(bc, data / f"preds_{name}_v3.jsonl",
                               data / f"report_{name}_v3.json")

    print("\n=== Mine hardcases ===", flush=True)
    s = mine_hardcases(bc, data / "report_asr_llm_v3.json",
                       data / "hardcases_v3.jsonl",
                       variants_per_pattern=50)
    print(f"Produced {s['n_hardcases']} hardcases across {len(s['patterns'])} patterns")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
