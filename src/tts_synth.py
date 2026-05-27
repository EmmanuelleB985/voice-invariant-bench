"""TTS backend interface — supports XTTS-v2 (default), ElevenLabs, Piper.

Each backend is a class with `synth(text, out_path, speaker, speed)` and a
`name` attribute used to namespace output paths and provenance.

To add a backend:
  1. Subclass TTSBackend.
  2. Implement `synth()`.
  3. Register in `BACKENDS`.

The stack-decomposable benchmark design depends on running multiple TTS
backends side by side. Audio paths are namespaced as `{backend_name}/...`
so a single row can carry audio from multiple backends in one dataset.
"""
from __future__ import annotations
import hashlib
import json
import os
import random
from pathlib import Path


class TTSBackend:
    name: str = "base"
    requires_gpu: bool = False
    speakers: list[str] = []
    default_speed: float = 1.0

    def synth(self, text: str, out_path: Path,
              speaker: str, speed: float = 1.0) -> None:
        raise NotImplementedError


# ---------- XTTS-v2 (open, neural, multi-speaker) -----------------------------

class XTTSBackend(TTSBackend):
    name = "xtts_v2"
    requires_gpu = True
    speakers = [
        "Claribel Dervla", "Daisy Studious", "Gracie Wise",
        "Tammie Ema", "Alison Dietlinde", "Ana Florence",
        "Annmarie Nele", "Asya Anara", "Brenda Stern",
        "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
        "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie",
        "Andrew Chipper", "Badr Odhiambo", "Dionisio Schuyler",
        "Royston Min", "Viktor Eka",
    ]

    def __init__(self):
        self._tts = None

    def _get(self):
        if self._tts is None:
            from TTS.api import TTS
            self._tts = TTS(
                "tts_models/multilingual/multi-dataset/xtts_v2",
                gpu=True, progress_bar=False,
            )
        return self._tts

    def synth(self, text, out_path, speaker, speed=1.0):
        out_path.parent.mkdir(exist_ok=True, parents=True)
        self._get().tts_to_file(
            text=text, speaker=speaker, language="en",
            file_path=str(out_path), speed=speed,
        )


# ---------- ElevenLabs (proprietary, neural, REST API) -----------------------

class ElevenLabsBackend(TTSBackend):
    name = "eleven"
    requires_gpu = False
    # Default voice IDs from ElevenLabs' shared library. These are stable.
    speakers = [
        "21m00Tcm4TlvDq8ikWAM",   # Rachel
        "AZnzlk1XvdvUeBnXmlld",   # Domi
        "EXAVITQu4vr4xnSDxMaL",   # Bella
        "ErXwobaYiN019PkySvjV",   # Antoni
        "VR6AewLTigWG4xSOukaG",   # Arnold
    ]

    def __init__(self, model_id: str = "eleven_turbo_v2_5"):
        self.model_id = model_id
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")

    def synth(self, text, out_path, speaker, speed=1.0):
        if not self.api_key:
            raise RuntimeError("ELEVENLABS_API_KEY not set")
        import requests
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{speaker}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        }
        out_path.parent.mkdir(exist_ok=True, parents=True)
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        # ElevenLabs returns mp3 by default. Save with .mp3 extension and a
        # parallel .wav symlink so file-existence checks work uniformly.
        mp3_path = out_path.with_suffix(".mp3")
        mp3_path.write_bytes(r.content)
        if out_path.suffix == ".wav" and not out_path.exists():
            out_path.symlink_to(mp3_path.name)


# ---------- Piper (open, lightweight, ONNX) -----------------------------------

class PiperBackend(TTSBackend):
    name = "piper"
    requires_gpu = False
    speakers = ["en_US-ryan-high", "en_US-amy-medium",
                "en_GB-alan-medium", "en_GB-jenny-medium"]

    def __init__(self, voices_dir: Path = Path("/workspace/piper_voices")):
        self.voices_dir = voices_dir

    def synth(self, text, out_path, speaker, speed=1.0):
        import subprocess
        out_path.parent.mkdir(exist_ok=True, parents=True)
        voice_path = self.voices_dir / f"{speaker}.onnx"
        if not voice_path.exists():
            raise RuntimeError(
                f"Piper voice {voice_path} not found. "
                "Download from rhasspy/piper releases.")
        subprocess.run(
            ["piper", "--model", str(voice_path),
             "--output_file", str(out_path),
             "--length_scale", str(1.0 / max(speed, 0.1))],
            input=text, text=True, check=True,
        )


# ---------- Dry-run (for offline tests) ---------------------------------------

class DryRunBackend(TTSBackend):
    name = "DRY_RUN"
    speakers = ["dryrun_speaker"]

    def synth(self, text, out_path, speaker, speed=1.0):
        out_path.parent.mkdir(exist_ok=True, parents=True)


# ---------- Registry ----------------------------------------------------------

BACKENDS: dict[str, type] = {
    "xtts_v2": XTTSBackend,
    "eleven": ElevenLabsBackend,
    "piper": PiperBackend,
    "DRY_RUN": DryRunBackend,
}


def get_backend(name: str) -> TTSBackend:
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown TTS backend '{name}'. Available: {list(BACKENDS)}"
        )
    return BACKENDS[name]()


# ---------- Pipeline helpers --------------------------------------------------

def _audio_path(row_id: str, turn_idx: int, text: str, backend_name: str) -> str:
    """Path for one turn's audio. Namespaced by backend so a single row can
    hold audio from multiple backends in parallel directories."""
    h = hashlib.md5(f"{row_id}_{turn_idx}_{text}".encode()).hexdigest()[:10]
    return f"{backend_name}/{row_id}/turn_{turn_idx:02d}_{h}.wav"


def synth_row_audio(row_dict: dict, out_dir: Path, rng: random.Random,
                    backend: TTSBackend, dry_run: bool = False) -> dict:
    """Synthesize all turns for one row with one backend.

    Result is added to row_dict["audio_dialogue"] (single-backend legacy)
    AND to row_dict["audio_dialogues"][backend.name] (multi-backend).
    Both forms are written for backward compatibility.
    """
    speaker = (random.Random().choice(backend.speakers)
               if backend.speakers else "default")
    # Use the passed rng deterministically for speaker selection
    speaker = (rng.choice(backend.speakers) if backend.speakers else "default")
    speed = rng.choice([0.85, 1.0, 1.1, 1.25])

    backend_audio = []
    for i, turn in enumerate(row_dict["reference_dialogue"]):
        rel = _audio_path(row_dict["id"], i, turn["text"], backend.name)
        full = out_dir / rel
        if not dry_run and not full.exists():
            backend.synth(turn["text"], full, speaker, speed=speed)
        backend_audio.append({
            "speaker": turn["speaker"],
            "text": turn["text"],
            "audio": rel,
        })

    row_dict["audio_dialogue"] = backend_audio
    row_dict.setdefault("audio_dialogues", {})[backend.name] = backend_audio

    prov = row_dict.setdefault("provenance", {})
    prov["tts_model"] = backend.name if not dry_run else "DRY_RUN"
    prov["tts_speaker"] = speaker
    prov["tts_speed"] = speed
    prov.setdefault("tts_backends_used", []).append(
        backend.name if not dry_run else "DRY_RUN"
    )
    return row_dict


def synthesize_jsonl(in_path: Path, out_jsonl: Path, audio_dir: Path,
                     seed: int = 0, limit: int | None = None,
                     dry_run: bool = False,
                     backend_name: str = "xtts_v2") -> None:
    """Synthesize one backend's audio for all rows."""
    rng = random.Random(seed)
    audio_dir.mkdir(exist_ok=True, parents=True)
    backend = DryRunBackend() if dry_run else get_backend(backend_name)
    n = 0
    with in_path.open() as fin, out_jsonl.open("w") as fout:
        for line in fin:
            if limit and n >= limit:
                break
            row = json.loads(line)
            row = synth_row_audio(row, audio_dir, rng, backend, dry_run=dry_run)
            fout.write(json.dumps(row) + "\n")
            n += 1
    mode = " (dry-run)" if dry_run else f" [{backend.name}]"
    print(f"TTS{mode}: synthesized {n} rows -> {audio_dir}")


def synthesize_jsonl_multi(in_path: Path, out_jsonl: Path, audio_dir: Path,
                            backend_names: list[str],
                            seed: int = 0,
                            limit: int | None = None) -> None:
    """Synthesize multiple backends' audio for the same rows in one pass.

    Each row ends up with audio_dialogues[backend_name] = [...] for every
    requested backend. Used for stack-decomposable benchmarking.
    """
    rng = random.Random(seed)
    audio_dir.mkdir(exist_ok=True, parents=True)
    backends = [get_backend(b) for b in backend_names]
    n = 0
    with in_path.open() as fin, out_jsonl.open("w") as fout:
        for line in fin:
            if limit and n >= limit:
                break
            row = json.loads(line)
            for backend in backends:
                row = synth_row_audio(row, audio_dir, rng, backend, dry_run=False)
            fout.write(json.dumps(row) + "\n")
            n += 1
    print(f"TTS multi [{', '.join(backend_names)}]: "
          f"synthesized {n} rows -> {audio_dir}")
