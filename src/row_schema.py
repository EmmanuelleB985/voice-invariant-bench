"""Canonical BenchmarkRow schema.

Wraps the invariant graph, dialogue plan, and provenance as first-class
objects. This is the single row type used everywhere in the pipeline.
"""
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field
import uuid

from src.invariant_graph import InvariantGraph
from src.dialogue_plan import DialoguePlan
from src.provenance import Provenance


Domain = Literal[
    "calendar", "retail", "travel", "support",
    "synthetic_transfer", "reminder", "forms",
]


class DialogueTurn(BaseModel):
    speaker: Literal["user", "agent"]
    text: str
    audio: Optional[str] = None


class ToolCall(BaseModel):
    tool: str
    arguments: dict


class BenchmarkRow(BaseModel):
    # Identity
    id: str = Field(default_factory=lambda: f"vib_{uuid.uuid4().hex[:12]}")
    domain: Domain
    task_type: str  # plan.schema_id by default
    risk_level: Literal["low", "medium", "high", "irreversible"]
    tool_irreversible: bool = False

    # Semantic core
    invariant_graph: InvariantGraph
    dialogue_plan: DialoguePlan

    # Rendered surface
    reference_dialogue: list[DialogueTurn]
    audio_dialogue: list[DialogueTurn] = []

    # Tool / state
    tool_schema: str
    initial_state: dict
    expected_tool_calls: list[ToolCall]
    expected_final_state: dict

    # Counterfactual linkage
    counterfactual_partner_id: Optional[str] = None
    counterfactual_changed_invariant_id: Optional[str] = None

    # Audit / provenance
    provenance: Provenance
    asr_audit: Optional[dict] = None
