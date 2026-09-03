"""First-party ultrasound image-quality metrics used by UGNS.

The generalized contrast-to-noise ratio follows A. Rodriguez-Molares et al.,
"The Generalized Contrast-to-Noise Ratio: A Formal Definition for Lesion
Detectability," IEEE TUFFC, 2020.
"""

from __future__ import annotations

import numpy as np


def _finite_1d(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return array[np.isfinite(array)]


def gcnr(region_a, region_b, bins: int = 256) -> float:
    """Return histogram-overlap gCNR as defined by Rodriguez-Molares et al."""
    a = _finite_1d(region_a)
    b = _finite_1d(region_b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    _, edges = np.histogram(np.concatenate((a, b)), bins=bins)
    hist_a, _ = np.histogram(a, bins=edges)
    hist_b, _ = np.histogram(b, bins=edges)
    prob_a = hist_a / max(float(hist_a.sum()), 1.0)
    prob_b = hist_b / max(float(hist_b.sum()), 1.0)
    return float(1.0 - np.minimum(prob_a, prob_b).sum())


def snr(region, eps: float = 1e-12) -> float:
    """Return the envelope signal-to-noise ratio, mean divided by std."""
    values = _finite_1d(region)
    if values.size == 0:
        return float("nan")
    return float(values.mean() / max(values.std(), eps))


def cnr(region_a, region_b, eps: float = 1e-12) -> float:
    """Return linear contrast-to-noise ratio for two regions."""
    a = _finite_1d(region_a)
    b = _finite_1d(region_b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    denominator = np.sqrt(a.var() + b.var())
    return float(abs(a.mean() - b.mean()) / max(denominator, eps))


def cnr_db(region_a, region_b, eps: float = 1e-12) -> float:
    """Return the dB CNR convention used for the UGNS paper evaluation."""
    a = _finite_1d(region_a)
    b = _finite_1d(region_b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    numerator = max(float(abs(a.mean() - b.mean()) ** 2), eps)
    denominator = max(float((a.var() + b.var()) / 2.0), eps)
    return float(10.0 * np.log10(numerator / denominator))


def contrast(region_a, region_b, eps: float = 1e-12) -> float:
    """Return mean-amplitude contrast between two regions in dB."""
    a = _finite_1d(region_a)
    b = _finite_1d(region_b)
    if a.size == 0 or b.size == 0:
        return float("nan")
    ratio = max(float(a.mean()), eps) / max(float(b.mean()), eps)
    return float(20.0 * np.log10(ratio))


def fwhm_6db(
    profile,
    *,
    log_compressed: bool = False,
    dynamic_range_db: float = 60.0,
    interpolation_factor: int = 10,
) -> float:
    """Measure the connected main-lobe width at minus 6 dB, in samples.

    ``log_compressed=True`` expects a display-normalized B-mode profile whose
    full scale spans ``dynamic_range_db``. Otherwise the profile is treated as
    a linear envelope amplitude.
    """
    values = np.asarray(profile, dtype=np.float64).ravel()
    if values.size < 2 or not np.any(np.isfinite(values)):
        return float("nan")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    peak = int(np.argmax(values))
    factor = max(1, int(interpolation_factor))
    x = np.arange(values.size, dtype=np.float64)
    x_interp = np.linspace(0.0, values.size - 1.0, values.size * factor)

    if log_compressed:
        relative_db = (values - values[peak]) * float(dynamic_range_db)
        interp = np.interp(x_interp, x, relative_db)
    else:
        peak_value = max(float(values[peak]), np.finfo(np.float64).eps)
        relative_db = 20.0 * np.log10(
            np.maximum(values, np.finfo(np.float64).eps) / peak_value
        )
        interp = np.interp(x_interp, x, relative_db)

    above = interp >= -6.0
    peak_interp = int(np.argmin(np.abs(x_interp - peak)))
    if not above[peak_interp]:
        return float("nan")
    left = peak_interp
    right = peak_interp
    while left > 0 and above[left - 1]:
        left -= 1
    while right + 1 < above.size and above[right + 1]:
        right += 1
    return float(x_interp[right] - x_interp[left])
