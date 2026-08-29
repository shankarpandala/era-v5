"""Task-6 invariants: the from-first-principles encoder vs torch, across formats,
tie cases, subnormals, and the canonical 0.1 answers."""
import struct
from fractions import Fraction

import pytest
import torch


def torch_bits(val, dtype, nbits):
    t = torch.tensor(val, dtype=torch.float32).to(dtype)
    iv = t.view({16: torch.int16, 8: torch.uint8}[nbits]).item() & ((1 << nbits) - 1)
    return f"{iv:0{nbits}b}", float(t)


def test_fraction_pitfall():
    assert Fraction(1, 10) != Fraction(0.1)


def test_canonical_tenth_bits(nb):
    bits32, stored32, _ = nb.encode_float(Fraction(1, 10), nb.FP32)
    assert bits32.replace("|", "") == "00111101110011001100110011001101"
    assert float(stored32) == struct.unpack(">f", struct.pack(">f", 0.1))[0]
    bits16, stored16, _ = nb.encode_float(Fraction(1, 10), nb.BF16)
    assert bits16.replace("|", "") == "0011110111001101"
    assert float(stored16) == 0.10009765625
    bits8, stored8, _ = nb.encode_float(Fraction(1, 10), nb.FP8E4M3)
    assert bits8.replace("|", "") == "00011101"
    assert float(stored8) == 0.1015625


@pytest.mark.parametrize("spec_name,dtype,nbits", [
    ("BF16", torch.bfloat16, 16),
    ("FP16", torch.float16, 16),
])
def test_encoder_matches_torch_random_sweep(nb, spec_name, dtype, nbits):
    """200 random float32 values: the exact-Fraction encoder must agree with
    torch's cast bit-for-bit (both round the same fp32 input, RNE)."""
    spec = getattr(nb, spec_name)
    g = torch.Generator().manual_seed(1310)
    vals = (torch.randn(200, generator=g) * torch.logspace(
        -8, 4, 200)).tolist()
    for v in vals:
        bits, stored, info = nb.encode_float(Fraction(v), spec)
        tb, tv = torch_bits(v, dtype, nbits)
        if info["overflow"]:
            continue    # torch saturates fp16/bf16 to inf; encoder flags instead
        assert bits.replace("|", "") == tb, f"{spec_name} {v}"
        assert float(stored) == tv, f"{spec_name} {v}"


def test_encoder_matches_torch_fp8(nb):
    g = torch.Generator().manual_seed(1311)
    vals = (torch.randn(200, generator=g) * torch.logspace(-4, 2, 200)).tolist()
    for v in vals:
        if abs(v) > 400:
            continue    # stay clear of the saturation edge
        bits, stored, info = nb.encode_float(Fraction(v), nb.FP8E4M3)
        tb, tv = torch_bits(v, torch.float8_e4m3fn, 8)
        assert bits.replace("|", "") == tb, f"fp8 {v}"
        assert float(stored) == tv, f"fp8 {v}"


def test_tie_to_even_both_directions(nb):
    for frac, spec, expected in [
            (Fraction(17, 256), nb.FP8E4M3, 0.0625),
            (Fraction(19, 256), nb.FP8E4M3, 0.078125),
            (Fraction(257, 256), nb.BF16, 1.0),
            (Fraction(259, 256), nb.BF16, 1.015625)]:
        _, stored, _ = nb.encode_float(frac, spec)
        assert float(stored) == expected


def test_fp16_subnormal_boundary(nb):
    _, s0, i0 = nb.encode_float(Fraction(1, 10 ** 8), nb.FP16)
    assert i0["underflow"] and float(s0) == 0.0
    _, s1, i1 = nb.encode_float(Fraction(3, 10 ** 8), nb.FP16)
    assert i1["subnormal"] and float(s1) == 2 ** -24
    assert float(torch.tensor(2.9e-8, dtype=torch.float32).to(torch.float16)) == 0.0
    assert float(torch.tensor(1e-8, dtype=torch.float32).to(torch.float16)) == 0.0
    assert float(torch.tensor(1e-8, dtype=torch.float32).to(torch.bfloat16)) != 0.0


def test_e4m3fn_saturation_and_nan():
    assert float(torch.tensor(500.0).to(torch.float8_e4m3fn)) == 448.0
    nan_bits = torch.tensor(float("nan")).to(torch.float8_e4m3fn).view(torch.uint8)
    assert nan_bits.item() & 0x7F == 0x7F


def test_mantissa_carry(nb):
    """1.9999999 rounds up through the mantissa into the next exponent."""
    _, stored, _ = nb.encode_float(Fraction(255, 128) + Fraction(1, 10 ** 9), nb.FP8E4M3)
    assert float(stored) == 2.0
