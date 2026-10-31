"""Trellis codebook registry shared by preparation, planning, and kernels.

A ``trellis_t256`` tile stores 256 tail-biting codes whose 16-bit decode
windows are interpreted by one codebook. The codebook is a model-level
setting; three are defined:

- ``mcg``: multiplicative congruential decode (multiplier ``0xCBAC1FED``)
  with a lop3 mask/or into two added fp16 halves.
- ``lut_e4m3``: integer permutation of each decode window followed by a
  fixed E4M3 value table; defined for K2/K3/K4.
- ``lut_fp16``: integer permutation followed by a piecewise-linear FP16 value
  law; defined for uniform K5/K6.
- ``procedural_npk``: table-free integer decode to E4M3, defined for
  K2/K3/K4. The same state law as ``mcg`` (multiplier ``0xCBAC1FED``, lop3
  mask/or, two added fp16 halves) plus an alpha-2 compander, converted to
  E4M3 instead of widened to fp16.

``CODEBOOKS`` is the set a checkpoint may declare to the preparation path.
``procedural_npk`` is deliberately not in it: it is a W4A8 trellis-arm
selection (``MoEDynamicKernelBackend(trellis_codebook="procedural_npk")``),
not a payload the W4A16 preparation/execution path can decode, and letting a
checkpoint declare it there would route it to a decoder that does not
implement it.

This module is torch-free. Kernel modules embed the ids as compile-time
constants, so the ids participate in kernel cache keys.
"""

from __future__ import annotations

MCG = "mcg"
LUT_E4M3 = "lut_e4m3"
LUT_FP16 = "lut_fp16"
PROCEDURAL_NPK = "procedural_npk"

CODEBOOKS: tuple[str, ...] = (MCG, LUT_E4M3, LUT_FP16)

# The laws the W4A8 trellis arm decodes to E4M3 operands (the ``mcg`` and
# ``lut_fp16`` decoders produce fp16/bf16 operands for the W4A16 family).
W4A8_ARM_CODEBOOKS: tuple[str, ...] = (LUT_E4M3, PROCEDURAL_NPK)

MCG_MULTIPLIER = 0xCBAC1FED
CODEBOOK_SENTINELS: dict[int, str] = {MCG_MULTIPLIER: MCG}


def normalize_codebook(codebook: str | int) -> str:
    """Return the canonical codebook id for ``codebook``.

    Integers are checkpoint sentinels (the MCG multiplier); strings are
    matched case-insensitively against the canonical ids.
    """

    if isinstance(codebook, int):
        normalized = CODEBOOK_SENTINELS.get(int(codebook) & 0xFFFFFFFF)
        if normalized is None:
            raise ValueError(
                "unsupported trellis codebook sentinel "
                f"{int(codebook) & 0xFFFFFFFF:#010x}; expected MCG 0xcbac1fed"
            )
        return normalized
    text = str(codebook).strip().lower()
    if text in CODEBOOKS:
        return text
    raise ValueError(
        f"unsupported trellis codebook {codebook!r}; expected "
        "'mcg', 'lut_e4m3', or 'lut_fp16'"
    )


def normalize_w4a8_arm_codebook(codebook: str) -> str:
    """Return the canonical id of a W4A8 trellis-arm decode law.

    ``lut_e4m3`` (default) and ``procedural_npk`` are the laws that decode a
    t256 ring window to an E4M3 operand byte for the W4A8 MMA. The ``mcg`` and
    ``lut_fp16`` ids name the W4A16 widening decoders, whose output is
    fp16/bf16, so they are rejected here rather than silently selecting a
    different arithmetic.
    """

    text = str(codebook).strip().lower()
    if text in W4A8_ARM_CODEBOOKS:
        return text
    raise ValueError(
        f"unsupported trellis_codebook {codebook!r}; the W4A8 trellis arm "
        f"decodes {W4A8_ARM_CODEBOOKS[0]!r} or {W4A8_ARM_CODEBOOKS[1]!r}"
    )


def validate_codebook_bits(codebook: str, bits: int) -> None:
    """Reject (codebook, bitrate) pairs the decoders do not define.

    MCG decodes any supported tile bitrate; the lookup-table codebooks are
    defined only on their construction ranges.
    """

    if codebook == LUT_E4M3 and bits not in (2, 3, 4):
        raise ValueError("lut_e4m3 is defined only for K2/K3/K4")
    if codebook == LUT_FP16 and bits not in (5, 6):
        raise ValueError("lut_fp16 is defined only for uniform K5/K6")
    if codebook == PROCEDURAL_NPK and bits not in (2, 3, 4):
        raise ValueError("procedural_npk is defined only for K2/K3/K4")
