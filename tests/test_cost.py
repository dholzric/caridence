# tests/test_cost.py
from caridence.bench.cost import (
    inspection_cost_api, inspection_cost_selfhosted, PRICES,
)


def test_api_cost_scales_with_frames():
    one = inspection_cost_api("gpt-4o", n_frames=1, in_tokens_per_frame=1000, out_tokens_per_frame=100)
    ten = inspection_cost_api("gpt-4o", n_frames=10, in_tokens_per_frame=1000, out_tokens_per_frame=100)
    assert ten > one
    assert abs(ten - 10 * one) < 1e-9


def test_known_price_math():
    # gpt-4o priced per PRICES table; cost = frames*(in/1e6*pin + out/1e6*pout)
    pin, pout = PRICES["gpt-4o"]
    cost = inspection_cost_api("gpt-4o", n_frames=2, in_tokens_per_frame=1_000_000, out_tokens_per_frame=0)
    assert abs(cost - 2 * pin) < 1e-6


def test_selfhosted_cheaper_than_api():
    api = inspection_cost_api("gpt-4o", n_frames=30, in_tokens_per_frame=1200, out_tokens_per_frame=120)
    sh = inspection_cost_selfhosted(gpu_hourly_usd=2.0, frames_per_sec=5.0, n_frames=30)
    assert sh < api
