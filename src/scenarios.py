"""Scenarios: the catalog of (domain, tool, invariants) the pipeline
generates from. Each scenario is small and explicit so coverage is auditable.

To add a domain: write invariant factories, add a scenario dict, write a
tool schema in tools/, done.
"""
from __future__ import annotations
import random
from src.invariant_graph import Invariant
from src.canonicalize import int_to_words, int_to_spelled_digits


# ---------- Invariant factories ----------------------------------------------

def make_street_number(rng: random.Random,
                       force_confusable: bool = False) -> Invariant:
    confusables = [13, 14, 15, 16, 17, 18, 19,
                   30, 40, 50, 60, 70, 80, 90]
    if force_confusable or rng.random() < 0.4:
        n = rng.choice(confusables)
        ambig = "phonetic_confusable"
        # 14 <-> 40, 15 <-> 50, etc.
        confusable_with = (n - 13 + 30) if n < 20 else (n // 10 - 3 + 10)
    else:
        n = rng.randint(1, 999)
        ambig = "none"
        confusable_with = None
    forms = [int_to_words(n)]
    if n >= 10:
        forms.append(int_to_spelled_digits(n))
    forms.append(str(n))
    return Invariant(
        type="street_number", surface_forms=forms, canonical_value=n,
        target_fields=["update_delivery_address.street_number",
                       "final_state.street_number"],
        ambiguity_class=ambig,
        confusable_with=confusable_with,
        acoustic_risk="medium" if ambig != "none" else "low",
    )


def make_amount(rng: random.Random,
                force_decimal_shift: bool = False) -> Invariant:
    if force_decimal_shift or rng.random() < 0.3:
        n = rng.choice([5.0, 50.0, 500.0, 0.5, 5000.0, 15.0, 150.0])
        ambig = "decimal_shift"
    else:
        n = round(rng.uniform(5, 5000), 2)
        ambig = "none"
    forms = [f"${n}", (f"{int_to_words(int(n))} dollars"
                       if n == int(n) else f"{n} dollars")]
    return Invariant(
        type="amount", surface_forms=forms, canonical_value=float(n),
        target_fields=["transfer_value.amount",
                       "final_state.last_transfer_amount",
                       "refund_order.amount",
                       "final_state.refund_amount"],
        ambiguity_class=ambig,
        acoustic_risk="high" if ambig != "none" else "medium",
    )


def make_code(rng: random.Random) -> Invariant:
    letters1 = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
    digits = "".join(rng.choices("0123456789", k=4))
    letters2 = rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
    code = f"{letters1}-{digits}-{letters2}"
    spoken = []
    for ch in code:
        if ch == "-":
            spoken.append("dash")
        elif ch.isalpha():
            spoken.append(ch)
        else:
            spoken.append(int_to_words(int(ch)))
    return Invariant(
        type="code", surface_forms=[code, " ".join(spoken)],
        canonical_value=code,
        target_fields=["transfer_value.recipient_id",
                       "final_state.last_transfer_recipient"],
        ambiguity_class="spelling_dependent",
        acoustic_risk="high",
    )


def make_time(rng: random.Random) -> Invariant:
    h = rng.randint(1, 12)
    m = rng.choice([0, 15, 30, 45])
    ampm = rng.choice(["a.m.", "p.m."])
    h24 = h % 12 + (12 if ampm == "p.m." else 0)
    canonical = f"{h24:02d}:{m:02d}"
    spoken = (f"{int_to_words(h)} o'clock {ampm}" if m == 0
              else f"{int_to_words(h)} {int_to_words(m)} {ampm}")
    return Invariant(
        type="time", surface_forms=[spoken, f"{h}:{m:02d} {ampm}"],
        canonical_value=canonical,
        target_fields=["reschedule_appointment.time", "final_state.appt_time",
                       "create_event.time", "set_reminder.time"],
        ambiguity_class="ampm_ambiguous",
        acoustic_risk="medium",
    )


def make_postcode(rng: random.Random) -> Invariant:
    area = "".join(rng.choices("ABCDEFGHIJKLMNOPRSTUWYZ",
                               k=rng.choice([1, 2])))
    district = str(rng.randint(1, 99))
    sector = str(rng.randint(0, 9))
    unit = "".join(rng.choices("ABCDEFGHJKLNPQRSTUVWXYZ", k=2))
    pc = f"{area}{district} {sector}{unit}"
    spoken = []
    for ch in pc.replace(" ", ""):
        if ch.isalpha():
            spoken.append(ch)
        else:
            spoken.append(int_to_words(int(ch)))
    return Invariant(
        type="postcode", surface_forms=[pc, " ".join(spoken)],
        canonical_value=pc,
        target_fields=["update_delivery_address.postcode",
                       "final_state.postcode"],
        ambiguity_class="spelling_dependent",
        acoustic_risk="high",
    )


def make_quantity(rng: random.Random) -> Invariant:
    n = rng.randint(1, 12)
    return Invariant(
        type="quantity",
        surface_forms=[int_to_words(n), str(n)],
        canonical_value=n,
        target_fields=["set_reminder.quantity",
                       "final_state.reminder_quantity"],
        ambiguity_class="none",
        acoustic_risk="low",
    )


# ---------- Scenarios --------------------------------------------------------

SCENARIOS = [
    # Retail: address
    {
        "domain": "retail",
        "tool_schema": "update_delivery_address",
        "tool_irreversible": False,
        "risk_level": "low",
        "invariant_factory": lambda rng: [make_street_number(rng)],
        "initial_state": {"street_number": 40, "street_name": "Westfield Road"},
        "make_args": lambda invs: {
            "order_id": "ORD-1001",
            "street_number": invs[0].canonical_value,
            "street_name": "Westfield Road",
        },
        "make_final": lambda invs: {
            "street_number": invs[0].canonical_value,
            "street_name": "Westfield Road",
        },
    },
    # Retail: postcode
    {
        "domain": "retail",
        "tool_schema": "update_postcode",
        "tool_irreversible": False,
        "risk_level": "low",
        "invariant_factory": lambda rng: [make_postcode(rng)],
        "initial_state": {"postcode": "OX1 1AA"},
        "make_args": lambda invs: {
            "order_id": "ORD-1001",
            "postcode": invs[0].canonical_value,
        },
        "make_final": lambda invs: {"postcode": invs[0].canonical_value},
    },
    # Calendar: reschedule
    {
        "domain": "calendar",
        "tool_schema": "reschedule_appointment",
        "tool_irreversible": False,
        "risk_level": "medium",
        "invariant_factory": lambda rng: [make_time(rng)],
        "initial_state": {"appt_id": "APPT-22", "appt_time": "09:00"},
        "make_args": lambda invs: {
            "appointment_id": "APPT-22",
            "date": "2026-06-01",
            "time": invs[0].canonical_value,
        },
        "make_final": lambda invs: {
            "appt_time": invs[0].canonical_value,
            "appt_date": "2026-06-01",
        },
    },
    # Calendar: create event
    {
        "domain": "calendar",
        "tool_schema": "create_event",
        "tool_irreversible": False,
        "risk_level": "medium",
        "invariant_factory": lambda rng: [make_time(rng)],
        "initial_state": {},
        "make_args": lambda invs: {
            "title": "Meeting",
            "date": "2026-06-01",
            "time": invs[0].canonical_value,
        },
        "make_final": lambda invs: {
            "event_created": True,
            "event_time": invs[0].canonical_value,
            "event_date": "2026-06-01",
        },
    },
    # Synthetic transfer — high-stakes
    {
        "domain": "synthetic_transfer",
        "tool_schema": "transfer_value",
        "tool_irreversible": True,
        "risk_level": "irreversible",
        "invariant_factory": lambda rng: [make_amount(rng), make_code(rng)],
        "initial_state": {"balance": 10000},
        "make_args": lambda invs: {
            "amount": next(i.canonical_value for i in invs if i.type == "amount"),
            "recipient_id": next(i.canonical_value for i in invs if i.type == "code"),
        },
        "make_final": lambda invs: {
            "last_transfer_amount": next(
                i.canonical_value for i in invs if i.type == "amount"),
            "last_transfer_recipient": next(
                i.canonical_value for i in invs if i.type == "code"),
        },
    },
    # Reminder
    {
        "domain": "reminder",
        "tool_schema": "set_reminder",
        "tool_irreversible": False,
        "risk_level": "medium",
        "invariant_factory": lambda rng: [make_quantity(rng), make_time(rng)],
        "initial_state": {},
        "make_args": lambda invs: {
            "quantity": next(i.canonical_value for i in invs if i.type == "quantity"),
            "time": next(i.canonical_value for i in invs if i.type == "time"),
        },
        "make_final": lambda invs: {
            "reminder_quantity": next(
                i.canonical_value for i in invs if i.type == "quantity"),
            "reminder_time": next(
                i.canonical_value for i in invs if i.type == "time"),
        },
    },
]
