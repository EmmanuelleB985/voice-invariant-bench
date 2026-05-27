"""Dialogue RENDERER — convert DialoguePlans into natural language.

Reads a plan + invariant graph and produces the spoken turns. The renderer
is the only part of the pipeline that knows about phrasing, so swapping it
out enables multilingual variants, register variations (formal/casual),
and persona variants without touching dialogue structure.

Renderers must be deterministic given (plan, graph, seed). The renderer
also appends LineageEvents to the graph as a side effect, so by the time
rendering finishes every invariant carries a full per-turn history.
"""
from __future__ import annotations
import random
from src.dialogue_plan import DialoguePlan, PlannedTurn
from src.invariant_graph import InvariantGraph, LineageEvent
from src.canonicalize import int_to_words, int_to_spelled_digits, time_to_spoken


def _spoken_form_for(inv_type: str, value) -> str:
    """Turn a canonical value back into a natural spoken surface form
    (e.g. 14 -> 'fourteen', '21:00' -> 'nine o'clock p.m.')."""
    if inv_type in ("street_number", "quantity") and isinstance(value, int):
        return int_to_words(value)
    if inv_type == "amount":
        try:
            return f"${float(value)}"
        except (TypeError, ValueError):
            return str(value)
    if inv_type == "time" and isinstance(value, str) and ":" in value:
        try:
            h, m = value.split(":")
            return time_to_spoken(int(h), int(m))
        except (ValueError, TypeError):
            return value
    return str(value)



# ---------- Template banks ---------------------------------------------------
# Multiple templates per (intent, schema_context). Add freely; renderer is the
# cheapest place to grow diversity.

USER_OPEN = {
    "update_delivery_address": [
        "Change my delivery to {sf} Westfield Road.",
        "Please update my delivery address to number {sf} Westfield Road.",
        "I need delivery to {sf} Westfield Road, please.",
    ],
    "update_postcode": [
        "Update my postcode to {sf}.",
        "Change the postcode on file to {sf}.",
    ],
    "reschedule_appointment": [
        "Move my appointment to {sf}.",
        "Reschedule me to {sf} please.",
        "I'd like to switch to {sf}.",
    ],
    "create_event": [
        "Book a meeting at {sf}.",
        "Schedule something for {sf}.",
    ],
    "transfer_value": [
        "Send {sf_amount} to {sf_code}.",
        "Please transfer {sf_amount} to recipient {sf_code}.",
        "I want to send {sf_amount} to {sf_code}.",
    ],
    "set_reminder": [
        "Remind me to take {sf_quantity} pills at {sf_time}.",
        "Set a reminder for {sf_quantity} pills, {sf_time}.",
    ],
}

AGENT_ECHO = [
    "Just to confirm — {echo}?",
    "{echo} — is that correct?",
    "So that's {echo}, right?",
    "Did I get that — {echo}?",
]

AGENT_CLARIFY = [
    "Sorry, could you repeat that?",
    "Could you spell that out for me?",
    "I want to make sure I got that right — could you say it again?",
]

USER_CORRECT_INT = [
    "No, I said {word_form}. {digit_form}.",
    "Not {wrong}, {right}.",
    "{right}, not {wrong}.",
]

USER_CORRECT_GENERIC = [
    "No, I said {right}, not {wrong}.",
    "Sorry, it's {right}, not {wrong}.",
]

USER_CONFIRM = [
    "Yes, that's right.", "Yes, confirm.", "Correct.",
    "That's the one.", "Yes please.",
]

USER_SELF_CORRECT = [
    "{slip} — wait, sorry, {right}.",
    "Make that {slip} — actually no, {right}.",
]


class DialogueRenderer:
    def __init__(self, tool_schema: str, seed: int = 0):
        self.tool_schema = tool_schema
        self.rng = random.Random(seed)

    def _pick_user_open(self) -> str:
        bank = USER_OPEN.get(self.tool_schema)
        if not bank:
            return "Please proceed with {sf}."
        return self.rng.choice(bank)

    def _add_inv_to_subs(self, inv, value, subs: dict[str, str],
                         is_primary: bool) -> None:
        """Write surface keys for this invariant into subs.

        is_primary=True means this invariant is in the turn's invariant_refs
        and gets to claim the generic keys (sf/echo/right). Secondary
        invariants only get their type-specific keys (sf_time, sf_amount, etc.)
        so reminder-style templates with multiple slots resolve cleanly.
        """
        sval = str(value)
        type_key = {
            "amount": "sf_amount",
            "code": "sf_code", "recipient_id": "sf_code", "order_id": "sf_code",
            "quantity": "sf_quantity",
            "time": "sf_time",
            "date": "sf_date",
            "postcode": "sf_postcode",
        }.get(inv.type)
        if type_key:
            subs.setdefault(type_key, sval)
        if is_primary:
            subs.setdefault("sf", sval)
            subs.setdefault("echo", sval)
            subs.setdefault("right", sval)

    def _spoken_value(self, inv, override) -> str:
        if override is not None:
            return _spoken_form_for(inv.type, override)
        forms = inv.surface_forms
        spoken = [
            f for f in forms
            if not f.replace("-", "").replace(".", "")
                  .replace("$", "").replace(" ", "").isdigit()
        ]
        return (spoken or forms)[0] if forms else str(inv.canonical_value)

    def _surface_for_turn(self, turn: PlannedTurn,
                          graph: InvariantGraph) -> dict[str, str]:
        subs: dict[str, str] = {}
        # Primary pass: invariants explicitly referenced by this turn
        referenced_ids = set(turn.invariant_refs)
        for inv_id in turn.invariant_refs:
            inv = graph.by_id(inv_id)
            if inv is None:
                continue
            value = self._spoken_value(inv, turn.value_override)
            self._add_inv_to_subs(inv, value, subs, is_primary=True)
        # Secondary pass: every other invariant in the graph, so templates
        # with multi-slot scenarios (e.g. reminder = quantity + time) resolve.
        for inv in graph.invariants:
            if inv.id in referenced_ids:
                continue
            value = self._spoken_value(inv, None)
            self._add_inv_to_subs(inv, value, subs, is_primary=False)
        return subs

    def render_turn(self, turn: PlannedTurn, graph: InvariantGraph,
                    plan: DialoguePlan) -> str:
        subs = self._surface_for_turn(turn, graph)

        if turn.role == "user" and turn.intent == "open_request":
            tmpl = self._pick_user_open()
            return _safe_format(tmpl, subs)

        if turn.role == "user" and turn.intent == "confirm":
            return self.rng.choice(USER_CONFIRM)

        if turn.role == "user" and turn.intent == "correct":
            inv = (graph.by_id(turn.invariant_refs[0])
                   if turn.invariant_refs else None)
            # In repair_mishear, the "wrong" value is what the agent said.
            # In conflicting_updates, the "wrong" value is the user's earlier
            # value, recorded in plan.metadata["original_value"].
            wrong = (plan.metadata.get("misheard_value")
                     or plan.metadata.get("original_value"))
            if wrong is None and inv is not None and inv.lineage:
                prior_introduce = next(
                    (e for e in inv.lineage if e.action == "introduce"), None,
                )
                if prior_introduce is not None:
                    wrong = prior_introduce.value_at_event
            if inv and isinstance(inv.canonical_value, int) and wrong is not None:
                wrong_str = (int_to_words(wrong)
                             if isinstance(wrong, int) else str(wrong))
                return self.rng.choice(USER_CORRECT_INT).format(
                    word_form=int_to_words(inv.canonical_value),
                    digit_form=int_to_spelled_digits(inv.canonical_value),
                    wrong=wrong_str,
                    right=int_to_words(inv.canonical_value),
                )
            wrong_str = _spoken_form_for(inv.type if inv else "", wrong) if wrong else ""
            return self.rng.choice(USER_CORRECT_GENERIC).format(
                right=subs.get("right", ""), wrong=wrong_str,
            )

        if turn.role == "user" and turn.intent == "self_correct":
            inv = (graph.by_id(turn.invariant_refs[0])
                   if turn.invariant_refs else None)
            slip = plan.metadata.get("slip_value")
            if inv and isinstance(inv.canonical_value, int):
                return self.rng.choice(USER_SELF_CORRECT).format(
                    slip=int_to_words(slip) if isinstance(slip, int) else str(slip),
                    right=int_to_words(inv.canonical_value),
                )
            # Non-int invariants: convert both slip and right to spoken forms.
            slip_spoken = (_spoken_form_for(inv.type, slip)
                           if inv is not None else str(slip))
            right_spoken = subs.get("sf", "")
            return self.rng.choice(USER_SELF_CORRECT).format(
                slip=slip_spoken, right=right_spoken,
            )

        if turn.role == "agent" and turn.intent in (
                "ask_confirmation", "echo_for_confirmation", "mishear_echo"):
            return _safe_format(self.rng.choice(AGENT_ECHO), subs)

        if turn.role == "agent" and turn.intent == "ask_clarification":
            return self.rng.choice(AGENT_CLARIFY)

        if turn.role == "agent" and turn.intent == "acknowledge":
            return "Got it."

        return f"[{turn.role}:{turn.intent}]"

    def render(self, plan: DialoguePlan, graph: InvariantGraph
               ) -> list[tuple[str, str]]:
        """Render plan into [(role, text), ...] and append lineage events."""
        out = []
        for turn in plan.turns:
            text = self.render_turn(turn, graph, plan)
            text = _normalize_punctuation(text)
            out.append((turn.role, text))
            for inv_id in turn.invariant_refs:
                inv = graph.by_id(inv_id)
                if inv is None:
                    continue
                action = _plan_action_to_lineage(turn)
                value = (turn.value_override
                         if turn.value_override is not None
                         else inv.canonical_value)
                inv.lineage.append(LineageEvent(
                    turn_idx=turn.turn_idx,
                    actor=turn.role,
                    action=action,
                    value_at_event=value,
                    surface_form_used=text[:80],
                ))
        return out


def _normalize_punctuation(text: str) -> str:
    """Collapse repeated terminal punctuation.

    XTTS-v2 doesn't care about '..' but downstream ASR and human readers do.
    Preserves intentional ellipsis ('...').
    """
    import re
    # ..  → .  (but preserve ellipsis ...)
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text)
    # ?. or ?.. → ?
    text = re.sub(r'\?\.+', '?', text)
    # !. or !.. → !
    text = re.sub(r'\!\.+', '!', text)
    return text


def _plan_action_to_lineage(turn: PlannedTurn):
    mapping = {
        "introduce": "introduce", "echo_correct": "echo",
        "echo_wrong": "mishear", "correct": "correct",
        "confirm": "confirm", "reject": "reject", "none": "echo",
    }
    return mapping.get(turn.action_on_invariant, "echo")


def _safe_format(tmpl: str, subs: dict) -> str:
    """str.format that doesn't crash on missing keys."""
    class _Default(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return tmpl.format_map(_Default(subs))
