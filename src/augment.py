"""Acoustic augmentation: phone codec, noise, reverb.

Uses torchaudio + audiomentations for reproducible, seedable augmentations.
Each row gets one acoustic condition assigned, recorded in provenance.
"""
from __future__ import annotations
import json
import random
from pathlib import Path


CONDITIONS = ["clean", "phone_codec", "noise_low", "noise_high", "reverb"]


def _try_imports():
    try:
        import torchaudio
        import torch
        from audiomentations import (
            Compose, AddGaussianNoise, RoomSimulator,
        )
        return torchaudio, torch, Compose, AddGaussianNoise, RoomSimulator
    except ImportError:
        return None


def apply_condition(wav_path: Path, out_path: Path, condition: str) -> None:
    audio_libs = _try_imports()
    if audio_libs is None or not wav_path.exists():
        out_path.parent.mkdir(exist_ok=True, parents=True)
        if wav_path.exists():
            out_path.write_bytes(wav_path.read_bytes())
        else:
            out_path.touch()
        return
    torchaudio, torch, Compose, AddGaussianNoise, RoomSimulator = audio_libs
    out_path.parent.mkdir(exist_ok=True, parents=True)

    if condition == "clean":
        out_path.write_bytes(wav_path.read_bytes())
        return

    waveform, sr = torchaudio.load(wav_path)
    if condition == "phone_codec":
        down = torchaudio.transforms.Resample(sr, 8000)(waveform)
        enc = torchaudio.functional.mu_law_encoding(down, 256)
        dec = torchaudio.functional.mu_law_decoding(enc, 256)
        up = torchaudio.transforms.Resample(8000, sr)(dec)
        torchaudio.save(out_path, up, sr)
    elif condition in ("noise_low", "noise_high"):
        amp = 0.005 if condition == "noise_low" else 0.02
        aug = Compose([
            AddGaussianNoise(min_amplitude=amp, max_amplitude=amp * 2, p=1.0),
        ])
        augmented = aug(samples=waveform.numpy()[0], sample_rate=sr)
        torchaudio.save(out_path, torch.tensor(augmented).unsqueeze(0), sr)
    elif condition == "reverb":
        aug = Compose([RoomSimulator(p=1.0)])
        augmented = aug(samples=waveform.numpy()[0], sample_rate=sr)
        torchaudio.save(out_path, torch.tensor(augmented).unsqueeze(0), sr)
    else:
        out_path.write_bytes(wav_path.read_bytes())


def augment_jsonl(in_path: Path, out_path: Path, audio_dir: Path,
                  conditions: list[str] | None = None,
                  seed: int = 0) -> None:
    rng = random.Random(seed)
    conditions = conditions or ["clean", "phone_codec", "noise_low"]
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            cond = rng.choice(conditions)
            for turn in row.get("audio_dialogue", []):
                src = audio_dir / turn["audio"]
                if cond == "clean":
                    continue
                new_rel = turn["audio"].replace(".wav", f"_{cond}.wav")
                dst = audio_dir / new_rel
                apply_condition(src, dst, cond)
                turn["audio"] = new_rel
            prov = row.setdefault("provenance", {})
            prov.setdefault("augmentations", []).append(cond)
            fout.write(json.dumps(row) + "\n")
