"""Provenance: every row carries enough metadata to be regenerated
or audited later. This is what makes the benchmark debuggable at scale.

Stored on the row itself (not in a sidecar) so it survives any
shuffle/filter/concatenate operation.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import subprocess


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()[:10]
    except Exception:
        return "unknown"


class Provenance(BaseModel):
    """Everything needed to re-derive or audit a row."""
    code_version: str = Field(default_factory=_git_sha)
    schema_version: str = "vib-1.0"
    created_at: str = Field(default_factory=_now)

    # Generation
    scenario_seed: int
    plan_seed: int
    render_seed: int
    plan_schema_id: str

    # Audio
    tts_model: str | None = None
    tts_speaker: str | None = None
    tts_speed: float | None = None
    augmentations: list[str] = []

    # ASR audit
    asr_model: str | None = None
    asr_version: str | None = None

    # Curation
    split: str | None = None
    curation_seed: int | None = None

    # Hardcase mining
    parent_row_id: str | None = None
    mined_from_failure: str | None = None

    notes: list[str] = []

    def add_note(self, note: str) -> None:
        self.notes.append(f"{_now()}: {note}")
