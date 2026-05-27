"""ASR backend interface — supports faster-whisper (default), Deepgram, AssemblyAI.

Each backend is a class with `transcribe(audio_path) -> str`. To add a backend:
  1. Subclass ASRBackend.
  2. Implement `transcribe()`.
  3. Register in `BACKENDS`.

For stack-decomposable benchmarking, you run multiple ASR backends over
each row's audio. Audit results are namespaced as
`asr_audits[backend_name] = {transcript, recovery, ...}` so the verifier
can compute per-(TTS, ASR) cell metrics.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from src.canonicalize import canonicalize


class ASRBackend:
    name: str = "base"
    requires_gpu: bool = False

    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError


# ---------- faster-whisper (open, batch, GPU) --------------------------------

class FasterWhisperBackend(ASRBackend):
    name = "whisper"
    requires_gpu = True

    def __init__(self, model_size: str = "large-v3"):
        self.model_size = model_size
        self._model = None

    def _get(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size, device="cuda", compute_type="float16",
            )
        return self._model

    def transcribe(self, audio_path):
        segments, _ = self._get().transcribe(
            str(audio_path), beam_size=5, vad_filter=True, language="en",
        )
        return " ".join(s.text.strip() for s in segments).strip()


# ---------- Deepgram (proprietary, streaming-capable, REST/SDK) --------------

class DeepgramBackend(ASRBackend):
    name = "deepgram"
    requires_gpu = False

    def __init__(self, model: str = "nova-3"):
        self.model = model
        self.api_key = os.environ.get("DEEPGRAM_API_KEY")

    def transcribe(self, audio_path):
        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set")
        import requests
        with audio_path.open("rb") as f:
            audio_bytes = f.read()
        url = f"https://api.deepgram.com/v1/listen?model={self.model}&smart_format=true"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }
        r = requests.post(url, data=audio_bytes, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        try:
            return data["results"]["channels"][0]["alternatives"][0]["transcript"]
        except (KeyError, IndexError):
            return ""


# ---------- AssemblyAI (proprietary, alternative error profile) --------------

class AssemblyAIBackend(ASRBackend):
    name = "assembly"
    requires_gpu = False

    def __init__(self, model: str = "universal"):
        self.model = model
        self.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

    def transcribe(self, audio_path):
        if not self.api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY not set")
        import requests, time
        headers = {"authorization": self.api_key}
        # Upload
        with audio_path.open("rb") as f:
            up = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=headers, data=f, timeout=60,
            )
        up.raise_for_status()
        upload_url = up.json()["upload_url"]
        # Submit transcription
        sub = requests.post(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers,
            json={"audio_url": upload_url, "speech_model": self.model},
            timeout=60,
        )
        sub.raise_for_status()
        tid = sub.json()["id"]
        # Poll
        for _ in range(60):
            r = requests.get(
                f"https://api.assemblyai.com/v2/transcript/{tid}",
                headers=headers, timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            if data["status"] == "completed":
                return data.get("text", "")
            if data["status"] == "error":
                return ""
            time.sleep(2)
        return ""


# ---------- Registry ----------------------------------------------------------

BACKENDS: dict[str, type] = {
    "whisper": FasterWhisperBackend,
    "deepgram": DeepgramBackend,
    "assembly": AssemblyAIBackend,
}


def get_backend(name: str) -> ASRBackend:
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown ASR backend '{name}'. Available: {list(BACKENDS)}"
        )
    return BACKENDS[name]()


# ---------- Recovery diagnostic ----------------------------------------------

def recover_invariants(transcript: str, expected: list[dict]) -> list[dict]:
    """For each expected invariant, attempt canonicalization from the transcript.

    Scan surface forms first, then try canonicalizing the whole transcript.
    Diagnostic only — not a substitute for SACR scoring.
    """
    out = []
    for inv in expected:
        recovered = None
        for form in inv["surface_forms"]:
            if form.lower() in transcript.lower():
                recovered = canonicalize(inv["type"], form)
                break
        if recovered is None:
            recovered = canonicalize(inv["type"], transcript)
        out.append({
            "type": inv["type"],
            "expected_canonical": inv["canonical_value"],
            "recovered_canonical": recovered,
            "match": recovered == inv["canonical_value"],
        })
    return out


# ---------- Pipeline ----------------------------------------------------------

def audit_jsonl(in_path: Path, audio_dir: Path, out_path: Path,
                backend_name: str = "whisper",
                tts_backend: str = "xtts_v2",
                limit: int | None = None) -> None:
    """Run ASR on every user turn for one TTS+ASR backend combination.

    Writes the result to row["asr_audits"][f"{tts_backend}__{asr_backend}"]
    AND to row["asr_audit"] (singular, legacy field).
    """
    backend = get_backend(backend_name)
    n = 0
    cell_key = f"{tts_backend}__{backend.name}"
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            if limit and n >= limit:
                break
            row = json.loads(line)
            audio_turns = (row.get("audio_dialogues", {}).get(tts_backend)
                           or row.get("audio_dialogue", []))
            user_audio = [t for t in audio_turns if t["speaker"] == "user"]
            transcripts = []
            for turn in user_audio:
                src = audio_dir / turn["audio"]
                transcripts.append(backend.transcribe(src) if src.exists() else "")
            full_transcript = " ".join(transcripts)
            invs = row.get("invariant_graph", {}).get("invariants", [])
            audit = recover_invariants(full_transcript, invs)
            audit_block = {
                "user_transcript": full_transcript,
                "invariant_recovery": audit,
                "any_failure": any(not a["match"] for a in audit),
            }
            row.setdefault("asr_audits", {})[cell_key] = audit_block
            row["asr_audit"] = audit_block   # legacy single-stack field
            prov = row.setdefault("provenance", {})
            prov["asr_model"] = backend.name
            prov.setdefault("asr_backends_used", []).append(cell_key)
            fout.write(json.dumps(row) + "\n")
            n += 1
    print(f"ASR audit [{cell_key}]: {n} rows")
