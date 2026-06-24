# Caridence — Vehicle Damage Walkaround Inspector

**Design spec · 2026-06-24**
**Event:** AMD Developer Hackathon: ACT II (lablab.ai) · Track 3 (Unicorn)
**Domain:** caridence.com ("car + evidence")
**Status:** design approved; pre-build phase begins before Jul 6 kickoff.

---

## 1. Summary

Caridence turns a 60-second phone walkaround of any vehicle into an objective,
structured damage report where **every finding cites the exact timestamped frame
it was seen in**, graded by severity with a repair-cost estimate. It produces
shareable, defensible proof of vehicle condition for rental returns, peer-to-peer
car sharing (Turo), dealer trade-ins, and insurance first-notice-of-loss.

The competitive moat — and the AMD story — is a **small open VLM fine-tuned and
served on the AMD MI300** that matches frontier (GPT-4o / Claude) detection
quality at roughly 1/30th the cost per inspection. We prove this live with a
benchmark dashboard in the demo.

## 2. Strategic rationale (why this wins)

The $10k pool is awarded to **top overall projects**, judged qualitatively on
**Application of Technology, Presentation, Business Value, Originality**. A single
design move scores on all four at once: a polished vertical product whose
centerpiece is a fine-tuned model on MI300 that provably matches frontier at a
fraction of the cost.

- **Application of Technology** → live MI300 fine-tune + vLLM serving + a measured
  benchmark, not an API wrapper.
- **Business Value** → "replace a $40 manual inspection / end deposit-style
  disputes with objective evidence" across rental, Turo, dealer, insurance.
- **Originality** → the video-walkaround → cited-frame UX in this vertical.
- **Presentation** → a dashboard showing cost ↓ while accuracy holds is a killer
  demo beat, and damage in cited frames reads instantly on camera.

Vehicle was chosen over rental/property/STR/construction because it is the only
vertical with abundant **public labeled data** (CarDD et al.), which is what makes
the "beats frontier at a fraction of the cost" benchmark credible rather than
hand-staged.

## 3. Foundational de-risking decisions

Driven by a prior hackathon where video issues blocked submission:

1. **No temporal video model.** Ingestion is `ffmpeg` frame sampling (1–2 fps) +
   optional Whisper audio transcript. Per-frame VLM, then aggregate. Video is a
   boring, reliable preprocessing step — not a research risk.
2. **Video is optional input.** The product accepts *either* a walkaround video
   *or* a folder of photos. If video processing hiccups during the official
   window, the demo still runs end-to-end on stills. Video can never sink us.
3. **Model tiering with no single point of failure.** The fine-tuned weights run
   on MI300 (production/moat), but the same weights fall back to a local 3090 or
   Fireworks if MI300 serving is flaky during the window.
4. **Mock mode.** A canned-report mode lets UI/report work proceed with no GPU or
   endpoint (proven pattern from prior QwenSight work).
5. **Cached benchmark results** so the dashboard always renders even if a baseline
   API is down mid-demo.
6. **Produced, AI-voiced demo** (no live narration) — UI and demo flow are
   designed around clean visual beats that need no live talking.

## 4. Architecture

### 4.1 Components (each a well-bounded unit)

| Unit | Responsibility | Key dependency |
|---|---|---|
| `ingest` | Video/photos → deduped, blur-filtered, timestamped frames (+ optional transcript) | ffmpeg, Whisper (opt) |
| `analyzer` | Per-frame VLM → structured detections `{panel, damage_type, severity, bbox, confidence}` | Qwen2.5-VL (local / MI300 / Fireworks) |
| `aggregator` | Merge multi-frame detections into unique findings; pick clearest "cited frame"; map to vehicle panel | pure Python |
| `estimator` | Finding → severity grade + repair-cost range (lookup table, part×type×severity) | static cost table |
| `report` | Assemble findings + cited frames + totals + condition score → JSON + HTML/PDF, shareable link | — |
| `web` | Upload/record UI, interactive report, benchmark/cost dashboard | FastAPI + light frontend |
| `bench` | Run our model vs frontier on held-out labeled set → accuracy + cost-per-inspection | the proof engine |
| `train` | Public datasets → detection/instruction format → LoRA fine-tune Qwen-VL → eval | PEFT; 3090s then MI300 |

### 4.2 Data flow

```
video/photos
   → ingest        (timestamped frames + optional transcript)
   → analyzer      (per-frame structured detections)
   → aggregator    (unique findings + best/cited frame + panel)
   → estimator     (severity + repair-cost range)
   → report        (JSON + HTML/PDF + shareable link)
   → web           (interactive report + dashboard)

bench: runs `analyzer` over a labeled held-out set offline → dashboard numbers
```

### 4.3 Model tiering

- **Dev/iteration:** Qwen2.5-VL on local 3090s + Fireworks.
- **Production / moat:** fine-tuned Qwen2.5-VL specialist served on **MI300 via vLLM**.
- **Benchmark baselines:** GPT-4o / Claude (frontier) + Qwen2.5-VL base (zero-shot).
- **Fallback:** fine-tuned weights also run on a 3090 or Fireworks.

## 5. The moat: fine-tune + benchmark

### 5.1 Why fine-tuning genuinely wins

Generalist VLMs are inconsistent at exactly what matters here: structured JSON
output, *not* hallucinating damage, calibrated severity grading, and consistent
panel naming. A specialist fine-tuned on car damage yields reliable JSON, fewer
false positives, and steadier severity → measurably higher detection F1 **and**
~30× cheaper because it is a small self-hosted model.

### 5.2 Base model & training target

- **Base:** Qwen2.5-VL (3B and 7B; both already in local HF cache). Headline the
  **7B** for accuracy; show **3B** as the "even cheaper" option.
- **Method:** LoRA on attention projections (proven recipe). QLoRA 4-bit +
  gradient checkpointing locally on 24 GB 3090s; full-fidelity run on MI300.
- **Grounding:** Qwen2.5-VL has native bbox output → fine-tune to emit bounding
  boxes, enabling the highlighted damage region on each cited frame.
- **Training target format:** `image → JSON [{damage_type, panel, severity, bbox:[x,y,w,h]}]`,
  strict-schema, prompt tokens masked.

### 5.3 Datasets

- **CarDD** (primary) — ~4,000 images, 6 damage classes (dent, scratch, crack,
  glass shatter, lamp broken, tire flat) with segmentation masks + bboxes.
- **Roboflow Universe** car-damage sets + a Kaggle COCO car-damage set — diversity.
- **Own van + Chevy footage** — hand-label a few dozen frames to close the
  domain gap between clean dataset images and real phone-walkaround frames
  (glare, motion blur, angles), and to make the demo vehicles in-distribution.

### 5.4 Label derivation pipeline

CarDD masks → bbox + damage_type directly; **panel** inferred by position
heuristic or VLM-assisted tagging; **severity** approximated from damage-area
ratio (small/moderate/severe), framed honestly as an estimate. Spot-check a
sample for quality.

### 5.5 Benchmark harness

- **Held-out test set:** CarDD test split + hand-labeled real car frames.
- **Contenders:** fine-tuned 7B (MI300) · fine-tuned 3B (MI300) · Qwen2.5-VL base
  zero-shot · GPT-4o · Claude.
- **Metrics:** per-damage-type precision/recall/F1, image-level "caught all
  damage" rate, false-positive (hallucinated damage) rate, **cost per inspection**
  (tokens×price vs ~free self-hosted), MI300 latency/throughput.
- **Output:** dashboard numbers + written report; results cached for demo safety.
- **Honesty rule:** claim only what the benchmark measures.

## 6. Error handling

- Blurry frames dropped via Laplacian-variance filter.
- Malformed VLM JSON → strict schema validation + retry (proven pattern).
- No damage found → valid "clean / proof-of-condition" report.
- Duplicate detections → aggregator dedupe by panel + type proximity across frames.
- Baseline API or MI300 down → cached benchmark results keep the dashboard live.

## 7. Testing

- Unit tests: aggregator dedupe, estimator lookup, JSON parser.
- Golden test: a small labeled set with known damage → assert finding count/types
  within tolerance.
- The `bench` harness doubles as an end-to-end integration test.

## 8. Stack & packaging

Python / FastAPI, ffmpeg, Whisper (optional), Qwen2.5-VL + vLLM, PEFT/LoRA, a
light dark-theme frontend. **Docker** containerization (a submission requirement).

## 9. Demo (produced, AI-voiced, ~2.5 min)

| Beat | Time | On screen | Voice |
|---|---|---|---|
| Hook | 0:00–0:20 | car b-roll; damage-dispute cost stat | the dispute every car sale/return/claim starts with |
| Intro | 0:20–0:35 | Caridence logo + tagline | "a 60-second walkaround becomes objective, cited proof" |
| Walkaround | 0:35–1:05 | real phone footage circling the Chevy → drag into app | "just walk around the car…" |
| Report reveal | 1:05–1:45 | 4 corner findings, each a cited frame w/ highlighted bbox, severity + $; total + condition score; click → zoom; cut to van-door major finding | "…finds every dent, scratch, and crack — pinned to the exact frame" |
| AMD moat | 1:45–2:20 | benchmark dashboard: fine-tuned 7B on MI300 vs GPT-4o — F1 parity, ~1/30th cost, MI300 latency/throughput | "our own model, fine-tuned and served on AMD MI300 — frontier accuracy at a fraction of the cost, self-hosted and private" |
| Business + close | 2:20–2:40 | markets (rental/Turo/dealer/insurance FNOL) + shareable proof link | "objective proof that ends the dispute" |

**Demo assets available:** a van with a damaged door (single major finding +
cost) and a Chevy with minor damage on all four corners (naturally 4 cited
findings). Submission assets: public GitHub repo, deployed demo URL, cover image,
slide deck, the video.

## 10. Timeline

**Phase 0 — Prep (Jun 24 – Jul 5, free local GPUs) → goal: fully working local demo before kickoff**
- Jun 24–26: repo + Docker scaffold; data pipeline (CarDD download + label→JSON
  conversion); `ingest` (ffmpeg frames); mock-mode UI + report renderer (full
  product shell exists).
- Jun 27–29: `analyzer` on base Qwen2.5-VL (zero-shot baseline); `aggregator` +
  `estimator` + `report` end-to-end on a real video; film van + Chevy, hand-label.
- Jun 30 – Jul 2: LoRA fine-tune 3B/7B on 3090s; `bench` harness; local
  fine-tuned-vs-base-vs-GPT-4o numbers.
- Jul 3–5: polish UI + dashboard, tighten accuracy, cut demo footage, draft slides
  + script. **Working local demo locked by Jul 5.**

**Phase 1 — Official window (Jul 6 – Jul 11, $100 MI300)**
- Jul 6: kickoff; spin up MI300; final fine-tune on MI300; serve via vLLM.
- Jul 7–8: run official benchmark on MI300; capture latency/throughput + cost.
- Jul 9–10: produce demo video (screen rec + phone footage + AI voice), slides,
  cover image, deploy demo URL, README + long description, verify container.
- Jul 11 AM: final review + submit early (deadline 11:00 AM CDT; target Jul 10 night).

## 11. Out of scope (YAGNI)

- True temporal video understanding.
- Pixel-perfect damage segmentation (rough bbox is enough for cited frames).
- Real-time mobile capture app (web upload/record is sufficient for the demo).
- Multi-tenant accounts / billing / auth beyond a preview gate.
- Verticals beyond vehicle (rental/property remain documented fallbacks).

## 12. Risks

| Risk | Mitigation |
|---|---|
| Domain gap: dataset images vs real walkaround frames | Augment with real van/Chevy frames + motion-blur/glare augmentation |
| Severity heuristic is crude | Label it "estimate"; sanity-check vs a few human judgments |
| 7B LoRA tight on 24 GB 3090 | QLoRA 4-bit + grad checkpointing locally; full run on MI300 |
| Video processing failure in demo | Photo-folder input path; mock mode; pre-rendered demo |
| Benchmark "beats frontier" claim doesn't hold | Claim only measured numbers; 3B/7B + cost angle still wins on cost even at parity-minus |
