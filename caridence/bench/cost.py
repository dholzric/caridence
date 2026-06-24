# caridence/bench/cost.py
from __future__ import annotations

# USD per 1M tokens (input, output). Update at bench time from current pricing.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    # self-hosted models priced via inspection_cost_selfhosted, not here
}


def inspection_cost_api(model: str, n_frames: int,
                        in_tokens_per_frame: int, out_tokens_per_frame: int) -> float:
    pin, pout = PRICES[model]
    per_frame = (in_tokens_per_frame / 1_000_000) * pin + (out_tokens_per_frame / 1_000_000) * pout
    return n_frames * per_frame


def inspection_cost_selfhosted(gpu_hourly_usd: float, frames_per_sec: float, n_frames: int) -> float:
    """Amortized GPU cost: (seconds of GPU time) * ($/sec)."""
    seconds = n_frames / frames_per_sec
    return seconds * (gpu_hourly_usd / 3600.0)
