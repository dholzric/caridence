# Caridence — Demo Video Script (AMD ACT II, Track 3: Unicorn)

Target length: ~3:15 at narration pace (~150 wpm). Narration is written to be read
by a TTS voice: short sentences, no parentheticals, numbers spelled the way they
should be spoken.

Legend: **[VISUAL]** = what's on screen while the line is spoken.

---

## Scene 1 — Hook (0:00–0:25)

**[VISUAL: phone-POV walkaround footage of the Chevy in the parking lot, cut fast]**

> Every used car sale, every rental return, and every insurance claim starts the
> same way. Someone walks around the car and writes down what they see.
> It's slow, it's inconsistent, and it misses things.

**[VISUAL: title card — Caridence logo, tagline]**

> This is Caridence. Point your phone at the car, walk around it once, and get a
> complete damage report. Every finding cited to the exact video frame, with
> severity and an estimated repair cost.

## Scene 2 — Live demo (0:25–1:20)

**[VISUAL: screen recording — upload 20260626_143914.mp4 (van) to the web app]**

> Here's a real walkaround, shot on a phone in one take. We upload it and
> Caridence goes to work.

**[VISUAL: report renders — findings list appears]**

> A few seconds later, the report. Caridence found the dent on the sliding door
> of this minivan, graded its severity, and estimated the repair cost.

**[VISUAL: click a finding → cited frame opens with bounding box]**

> Every finding links back to the exact frame and the exact box in the video.
> No hand-waving. You can audit every claim the system makes.

**[VISUAL: the Chevy report — multiple scratch findings across corners]**

> On this Chevy, it caught parking-lot scratches on multiple corners.
> And on the clean cars we tested, it stayed quiet. No damage means no findings.

**[VISUAL: plate redaction before/after (redact_demo/plate_before.jpg → plate_after.jpg)]**

> Privacy is built in. License plates are detected and blurred automatically
> before anything is stored or shared.

## Scene 3 — How it works (1:20–2:10)

**[VISUAL: pipeline diagram — video → frames → detector → VLM verify → report]**

> Under the hood, Caridence is a hybrid of two fine-tuned open models.
> A YOLO detector, fine-tuned on the CarDD damage dataset, scans every frame
> with very high recall. It reaches ninety-eight percent presence recall on the
> held-out test set.

**[VISUAL: detector candidates on a frame → crops → verifier yes/no overlay]**

> Then a fine-tuned Qwen vision language model inspects each candidate crop and
> confirms or rejects it. The detector guarantees we don't miss damage.
> The verifier keeps false alarms out of the report.

**[VISUAL: benchmark chart — base VLM F1 ~0 vs fine-tuned 0.50 / 0.56]**

> Fine-tuning is the whole story here. The same base model, zero-shot, scores
> almost zero on grounded damage detection. Our fine-tune takes it to a viable
> product. Small open models, specialized, running on our own hardware.

## Scene 4 — AMD platform (2:10–2:40)

**[VISUAL: terminal on the AMD ROCm box running scripts/bench_amd.py → the 66 FPS result; then docs/amd-benchmark.md]**

> The whole stack is open source and built for AMD. Training and serving run on
> ROCm. We benchmarked the detector live on an AMD Radeon GPU: sixty-six frames
> per second single pass, twenty-six with full test-time augmentation. A
> sixty-second walkaround is analyzed faster than it was filmed, and the same
> ROCm code path scales straight up to an Instinct MI300X. The verifier can run
> as Gemma through Fireworks AI on AMD hardware — one environment variable.

## Scene 5 — Market + close (2:40–3:15)

**[VISUAL: quick montage — rental counter, dealership lot, insurance app mock; then results site caridence.com/results]**

> Rental fleets inspect millions of returns a year. Dealers and marketplaces
> need condition reports they can trust. Insurers need claims triaged in
> minutes, not days. Caridence turns a sixty-second walk with a phone into a
> report all of them can rely on.

**[VISUAL: title card — Caridence, caridence.com, GitHub URL]**

> Caridence. Walk around the car. Get the truth.
> Built on open models and AMD infrastructure, for the AMD Developer Hackathon.

---

## Production notes

- Narration word count: ~430 words ≈ 3:00–3:20 at TTS pace.
- TTS: local Kokoro-82M; render per-scene WAVs so scenes can be re-cut without
  re-rendering everything.
- Screen recordings needed: upload→report flow (van + Chevy), finding-click →
  cited frame, redaction before/after, benchmark dashboard.
- Voice sample line (used for voice selection):
  "This is Caridence. Point your phone at the car, walk around it once, and get
  a complete damage report."
