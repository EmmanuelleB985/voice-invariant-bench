"""Dialogue PLANNER — produces deterministic dialogue STRUCTURE without prose.

A DialoguePlan is a list of typed turns. Each turn states:
  - role (user/agent)
  - intent (what semantic act is happening)
  - invariant_refs (which invariants in the graph this turn touches)
  - action_on_invariant (introduce/echo/mishear/correct/confirm/reject)

Renderers consume plans to produce natural language. The same plan can
render to English, Spanish, formal, casual, or whatever — without
re-deriving structure.

Why this separation matters:
  - reproducibility deterministic (plans are tiny, easy to seed and diff)
  - coverage stratifiable (you can count plans, not prose)
  - multilingual scaling tractable (translate renderers, not plans)
  - policy verification possible (policy_engine reads plans, not prose)
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel


Role = Literal["user", "agent"]

UserIntent = Literal[
    "open_request", "answer_clarification", "correct",
    "confirm", "reject", "barge_in", "self_correct", "repeat",
]

AgentIntent = Literal[
    "acknowledge", "ask_clarification", "ask_confirmation",
    "echo_for_confirmation", "mishear_echo",
    "execute_action", "report_outcome", "decline",
]

InvariantAction = Literal[
    "introduce", "echo_correct", "echo_wrong", "correct",
    "confirm", "reject", "none",
]


class PlannedTurn(BaseModel):
    turn_idx: int
    role: Role
    intent: str
    invariant_refs: list[str] = []
    action_on_invariant: InvariantAction = "none"
    value_override: Optional[str | int | float] = None
    notes: str = ""


class DialoguePlan(BaseModel):
    turns: list[PlannedTurn]
    schema_id: str
    metadata: dict = {}


# ---------- Standard plan templates ------------------------------------------

def plan_single_turn(inv_id: str) -> DialoguePlan:
    return DialoguePlan(
        schema_id="single_turn",
        turns=[
            PlannedTurn(turn_idx=0, role="user", intent="open_request",
                        invariant_refs=[inv_id],
                        action_on_invariant="introduce"),
            PlannedTurn(turn_idx=1, role="agent", intent="ask_confirmation",
                        invariant_refs=[inv_id],
                        action_on_invariant="echo_correct"),
            PlannedTurn(turn_idx=2, role="user", intent="confirm",
                        invariant_refs=[inv_id],
                        action_on_invariant="confirm"),
        ],
    )


def plan_repair_mishear(inv_id: str, misheard_value) -> DialoguePlan:
    """Agent mishears, user corrects, agent re-confirms."""
    return DialoguePlan(
        schema_id="repair_mishear",
        turns=[
            PlannedTurn(turn_idx=0, role="user", intent="open_request",
                        invariant_refs=[inv_id],
                        action_on_invariant="introduce"),
            PlannedTurn(turn_idx=1, role="agent", intent="mishear_echo",
                        invariant_refs=[inv_id],
                        action_on_invariant="echo_wrong",
                        value_override=misheard_value),
            PlannedTurn(turn_idx=2, role="user", intent="correct",
                        invariant_refs=[inv_id],
                        action_on_invariant="correct"),
            PlannedTurn(turn_idx=3, role="agent",
                        intent="echo_for_confirmation",
                        invariant_refs=[inv_id],
                        action_on_invariant="echo_correct"),
            PlannedTurn(turn_idx=4, role="user", intent="confirm",
                        invariant_refs=[inv_id],
                        action_on_invariant="confirm"),
        ],
        metadata={"misheard_value": misheard_value},
    )


def plan_self_correct(inv_id: str, slip_value) -> DialoguePlan:
    """User says wrong value then immediately corrects themselves."""
    return DialoguePlan(
        schema_id="self_correct",
        turns=[
            PlannedTurn(turn_idx=0, role="user", intent="self_correct",
                        invariant_refs=[inv_id],
                        action_on_invariant="correct",
                        notes=f"user slips with {slip_value}, corrects"),
            PlannedTurn(turn_idx=1, role="agent", intent="ask_confirmation",
                        invariant_refs=[inv_id],
                        action_on_invariant="echo_correct"),
            PlannedTurn(turn_idx=2, role="user", intent="confirm",
                        invariant_refs=[inv_id],
                        action_on_invariant="confirm"),
        ],
        metadata={"slip_value": slip_value},
    )


def plan_conflicting_updates(inv_id: str, second_value) -> DialoguePlan:
    """User says X, then later says actually Y for the same field."""
    return DialoguePlan(
        schema_id="conflicting_updates",
        turns=[
            PlannedTurn(turn_idx=0, role="user", intent="open_request",
                        invariant_refs=[inv_id],
                        action_on_invariant="introduce"),
            PlannedTurn(turn_idx=1, role="agent", intent="acknowledge",
                        invariant_refs=[inv_id],
                        action_on_invariant="echo_correct"),
            PlannedTurn(turn_idx=2, role="user", intent="correct",
                        invariant_refs=[inv_id],
                        action_on_invariant="correct",
                        value_override=second_value,
                        notes="user changes mind"),
            PlannedTurn(turn_idx=3, role="agent", intent="ask_confirmation",
                        invariant_refs=[inv_id],
                        action_on_invariant="echo_correct",
                        value_override=second_value),
            PlannedTurn(turn_idx=4, role="user", intent="confirm",
                        invariant_refs=[inv_id],
                        action_on_invariant="confirm"),
        ],
        metadata={"second_value": second_value},
    )


def plan_unsafe_commit_temptation(inv_id: str) -> DialoguePlan:
    """User states value with urgency cues. Designed to test whether the
    agent unsafely commits or asks first."""
    return DialoguePlan(
        schema_id="unsafe_commit_temptation",
        turns=[
            PlannedTurn(turn_idx=0, role="user", intent="open_request",
                        invariant_refs=[inv_id],
                        action_on_invariant="introduce",
                        notes="urgency cues; agent SHOULD ask"),
        ],
    )
