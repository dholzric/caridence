"""Render the full demo narration (docs/demo/demo-script.md) as per-scene WAVs.

Run with the TTS venv:  .venv-tts/Scripts/python scripts/tts_render.py
Outputs to demo_review/narration/scene_NN.wav (24 kHz mono).
Scenes are kept separate so the video can be re-cut without re-rendering.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

VOICE = "af_heart"
SAMPLE_RATE = 24000
PAUSE = np.zeros(int(0.55 * SAMPLE_RATE), dtype=np.float32)  # gap between paragraphs

SCENES = {
    "scene_01_hook": [
        "Every used car sale, every rental return, and every insurance claim "
        "starts the same way. Someone walks around the car and writes down what "
        "they see. It's slow, it's inconsistent, and it misses things.",
        "This is Caridence. Point your phone at the car, walk around it once, "
        "and get a complete damage report. Every finding cited to the exact "
        "video frame, with severity and an estimated repair cost.",
    ],
    "scene_02_demo": [
        "Here's a real walkaround, shot on a phone in one take. We upload it "
        "and Caridence goes to work.",
        "A few seconds later, the report. Caridence found the dent on the "
        "sliding door of this minivan, graded its severity, and estimated the "
        "repair cost.",
        "Every finding links back to the exact frame and the exact box in the "
        "video. No hand-waving. You can audit every claim the system makes.",
        "On this Chevy, it caught parking-lot scratches on multiple corners. "
        "And on the clean cars we tested, it stayed quiet. No damage means no "
        "findings.",
        "Privacy is built in. License plates are detected and blurred "
        "automatically before anything is stored or shared.",
    ],
    "scene_03_how": [
        "Under the hood, Caridence is a hybrid of two fine-tuned open models. "
        "A YOLO detector, fine-tuned on the CarDD damage dataset, scans every "
        "frame with very high recall. It reaches ninety-eight percent presence "
        "recall on the held-out test set.",
        "Then a fine-tuned Qwen vision language model inspects each candidate "
        "crop and confirms or rejects it. The detector guarantees we don't miss "
        "damage. The verifier keeps false alarms out of the report.",
        "Fine-tuning is the whole story here. The same base model, zero-shot, "
        "scores almost zero on grounded damage detection. Our fine-tune takes "
        "it to a viable product. Small open models, specialized, running on our "
        "own hardware.",
    ],
    "scene_04_amd": [
        "The whole stack is open source and built for AMD. Training and serving "
        "run on ROCm. We benchmarked the detector live on an AMD Radeon GPU: "
        "sixty-six frames per second single pass, twenty-six with full "
        "test-time augmentation. A sixty-second walkaround is analyzed faster "
        "than it was filmed, and the same ROCm code path scales straight up to "
        "an Instinct M I three hundred X. The verifier can run as Gemma through "
        "Fireworks AI on AMD hardware, with one environment variable.",
    ],
    "scene_05_close": [
        "Rental fleets inspect millions of returns a year. Dealers and "
        "marketplaces need condition reports they can trust. Insurers need "
        "claims triaged in minutes, not days. Caridence turns a sixty-second "
        "walk with a phone into a report all of them can rely on.",
        "Caridence. Walk around the car. Get the truth. Built on open models "
        "and AMD infrastructure, for the AMD Developer Hackathon.",
    ],
}

OUT_DIR = Path(__file__).resolve().parents[1] / "demo_review" / "narration"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = KPipeline(lang_code="a")
    total = 0.0
    for name, paragraphs in SCENES.items():
        parts = []
        for text in paragraphs:
            parts.extend(audio for _, _, audio in pipeline(text, voice=VOICE))
            parts.append(PAUSE)
        audio = np.concatenate(parts[:-1])  # drop trailing pause
        out = OUT_DIR / f"{name}.wav"
        sf.write(out, audio, SAMPLE_RATE)
        secs = len(audio) / SAMPLE_RATE
        total += secs
        print(f"wrote {out.name} ({secs:.1f}s)")
    print(f"total narration: {total:.1f}s")


if __name__ == "__main__":
    main()
