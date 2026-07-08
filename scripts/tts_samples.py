"""Render the demo-script sample line in several Kokoro voices for voice selection.

Run with the TTS venv:  .venv-tts/Scripts/python scripts/tts_samples.py
Outputs WAVs to demo_review/voice_samples/.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_LINE = (
    "This is Caridence. Point your phone at the car, walk around it once, "
    "and get a complete damage report. Every finding cited to the exact video "
    "frame, with severity and an estimated repair cost."
)

VOICES = [
    "af_heart",    # American female — Kokoro's flagship voice
    "af_bella",    # American female — warmer
    "af_nicole",   # American female — soft/whispery
    "am_michael",  # American male — even narrator
    "am_fenrir",   # American male — deeper
    "bm_george",   # British male — documentary feel
]

OUT_DIR = Path(__file__).resolve().parents[1] / "demo_review" / "voice_samples"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = KPipeline(lang_code="a")
    for voice in VOICES:
        chunks = [audio for _, _, audio in pipeline(SAMPLE_LINE, voice=voice)]
        audio = np.concatenate(chunks)
        out = OUT_DIR / f"{voice}.wav"
        sf.write(out, audio, 24000)
        print(f"wrote {out} ({len(audio) / 24000:.1f}s)")


if __name__ == "__main__":
    main()
