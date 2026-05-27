"""Invariant graph — the semantic core of VoiceInvariantBench.

Every spoken value lives in a graph with:
  - typed canonical value + surface forms
  - target_fields: which tool arguments / state fields it controls
  - dependencies: other invariants whose canonical value affects this one
  - lineage: who introduced this value, who corrected it, who confirmed it
  - injected-risk metadata (NOT pretend confidence priors)
  - criticality: scoring weight

The graph supports cross-turn repair, conflicting updates, confirmation
tracking, counterfactual tracing, and state-diff analysis as first-class
operations rather than ad-hoc string manipulation.
"""
from __future__ import annotations
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field
import uuid


InvariantType = Literal[
    "date", "time", "datetime", "duration",
    "amount", "quantity", "unit", "currency",
    "street_number", "street_name", "postcode", "city",
    "email", "url", "phone",
    "code", "recipient_id", "order_id", "ticket_id",
    "person_name",
]

Criticality = Literal["critical", "important", "informational"]
AmbiguityClass = Literal[
    "none",
    "phonetic_confusable",     # thirteen/thirty, fourteen/forty
    "decimal_shift",           # 50 vs 500 vs 5.0
    "ampm_ambiguous",          # "nine" alone
    "spelling_dependent",      # codes that need letter-by-letter
    "homophone",               # "to/two/too", "for/four"
    "polysemous_number",       # "two fifty" = 250 or 2:50?
]
AcousticRisk = Literal["low", "medium", "high"]


class LineageEvent(BaseModel):
    """One event in an invariant's history. Used to verify repair semantics."""
    turn_idx: int
    actor: Literal["user", "agent", "system"]
    action: Literal["introduce", "echo", "mishear", "correct",
                    "confirm", "reject", "query"]
    value_at_event: Any
    surface_form_used: Optional[str] = None
    notes: str = ""


class Invariant(BaseModel):
    id: str = Field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:8]}")
    type: InvariantType
    surface_forms: list[str]
    canonical_value: Any
    criticality: Criticality = "critical"

    # Which tool arguments / final-state fields this controls.
    # e.g. ["update_delivery_address.street_number", "final_state.street_number"]
    target_fields: list[str]

    dependencies: list[str] = []
    lineage: list[LineageEvent] = []

    # Risk factors we DELIBERATELY INJECTED. Not estimated probabilities.
    ambiguity_class: AmbiguityClass = "none"
    acoustic_risk: AcousticRisk = "low"
    confusable_with: Optional[Any] = None

    @property
    def was_corrected(self) -> bool:
        return any(e.action == "correct" for e in self.lineage)

    @property
    def was_confirmed(self) -> bool:
        return any(e.action == "confirm" for e in self.lineage)

    @property
    def final_value(self) -> Any:
        """The canonical value as of the last meaningful lineage event."""
        if not self.lineage:
            return self.canonical_value
        for ev in reversed(self.lineage):
            if ev.action in ("introduce", "correct"):
                return ev.value_at_event
        return self.canonical_value


class InvariantGraph(BaseModel):
    invariants: list[Invariant] = []

    def by_id(self, iid: str) -> Optional[Invariant]:
        return next((i for i in self.invariants if i.id == iid), None)

    def by_type(self, t: InvariantType) -> list[Invariant]:
        return [i for i in self.invariants if i.type == t]

    def critical(self) -> list[Invariant]:
        return [i for i in self.invariants if i.criticality == "critical"]

    def for_target(self, field_path: str) -> Optional[Invariant]:
        return next(
            (i for i in self.invariants if field_path in i.target_fields),
            None,
        )

    def diff(self, other: "InvariantGraph") -> dict[str, tuple]:
        """{inv_id: (self_value, other_value)} for differing invariants."""
        out = {}
        other_by_id = {i.id: i for i in other.invariants}
        for inv in self.invariants:
            o = other_by_id.get(inv.id)
            if o is None:
                continue
            if inv.canonical_value != o.canonical_value:
                out[inv.id] = (inv.canonical_value, o.canonical_value)
        return out

    def add_event(self, invariant_id: str, event: LineageEvent) -> None:
        inv = self.by_id(invariant_id)
        if inv:
            inv.lineage.append(event)

    def repair_targets(self) -> list[Invariant]:
        return [i for i in self.invariants if i.was_corrected]
