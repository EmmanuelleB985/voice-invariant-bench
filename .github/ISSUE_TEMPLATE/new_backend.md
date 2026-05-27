---
name: New backend (TTS or ASR)
about: Propose adding a new TTS or ASR backend to the benchmark
title: 'Backend: '
labels: enhancement, backend
assignees: ''

---

**Backend name and type** (e.g. "ElevenLabs TTS" or "Deepgram ASR")

**Why this backend matters**
What unique failure mode or quality does it expose that existing backends don't?

**API / library**
- Where docs live:
- License:
- Authentication: API key / OAuth / local install / other

**Cost estimate**
Per second of audio, or per character, whichever applies.

**Implementation plan**
- [ ] Class subclassing `TTSBackend` or `ASRBackend`
- [ ] Registered in `BACKENDS` dict
- [ ] Test that it imports + registers
- [ ] Smoke test against 5 rows
- [ ] Cost note added to `COSTS.md`

I am willing to implement this myself: yes / no
