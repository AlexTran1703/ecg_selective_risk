"""The three fixed signal representations: Raw, VCG and FFT.

All three are one-dimensional and go through the same 1-D backbone, differing
only in the number of input channels.  That keeps the benchmark a comparison of
*representations* rather than a comparison of architectures:

    Raw  12 x 5000   time axis        full temporal morphology
    VCG   3 x 5000   time axis        spatial cardiac vector (inverse Dower)
    FFT  12 x 496    frequency axis   global spectral content, phase discarded

Nothing here is tuned.  Each representation has exactly one configuration,
chosen up front and held constant across every pipeline and every
leave-one-source-out split, so the study compares representations rather than
preprocessing searches.

``stft_torch`` is retained as a 2-D alternative but is not part of the main
study: it would force a different (2-D) backbone and so confound representation
with architecture.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

BAND_HZ = (0.5, 50.0)
FILTER_ORDER = 3

# Inverse Dower transform (Edenbrandt & Pahlm): 8 independent leads -> X, Y, Z.
# Columns follow (I, II, V1, V2, V3, V4, V5, V6).
INVERSE_DOWER = np.array([
    [ 0.156, -0.010, -0.172, -0.074,  0.122,  0.231,  0.239,  0.194],  # X
    [-0.227,  0.887,  0.057, -0.019, -0.106, -0.022,  0.041,  0.048],  # Y
    [ 0.022,  0.102, -0.229, -0.310, -0.246, -0.063,  0.055,  0.108],  # Z
], dtype=np.float32)

# Row indices of the independent leads inside STANDARD_LEADS.
_INDEPENDENT_LEAD_ROWS = (0, 1, 6, 7, 8, 9, 10, 11)   # I, II, V1..V6

# Fixed STFT configuration (2-D; kept for reference, not used in the study).
N_FFT = 256
HOP = 64
STFT_BAND_HZ = BAND_HZ

# Fixed FFT configuration.  A 10 s window at 500 Hz gives df = 0.1 Hz, so the
# retained 0.5-50 Hz band is 496 bins per lead.
FFT_BAND_HZ = BAND_HZ


def bandpass_notch(signal: np.ndarray, fs: float, mains_hz: float) -> np.ndarray:
    """Minimal filtering: mains notch, then a 0.5-50 Hz zero-phase bandpass.

    Amplitude is left in millivolts -- QRS voltage is itself diagnostic, so no
    per-record amplitude normalisation happens here.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if mains_hz and mains_hz < fs / 2:
        b, a = iirnotch(w0=mains_hz / (fs / 2), Q=30.0)
        x = filtfilt(b, a, x, axis=-1)

    sos = butter(FILTER_ORDER, [BAND_HZ[0] / (fs / 2), BAND_HZ[1] / (fs / 2)],
                 btype="bandpass", output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    return np.ascontiguousarray(x, dtype=np.float32)


def to_vcg(signal12: np.ndarray) -> np.ndarray:
    """Map a 12-lead record in millivolts to the VCG ``(3, T)`` X/Y/Z leads.

    Applied to physical amplitudes *before* any normalisation, so the transform
    stays physically meaningful.
    """
    independent = signal12[list(_INDEPENDENT_LEAD_ROWS)]
    return (INVERSE_DOWER @ independent).astype(np.float32)


def fft_band_bins(n_samples: int = 5000, fs: float = 500.0,
                  band: tuple[float, float] = FFT_BAND_HZ) -> tuple[int, int]:
    """Inclusive rFFT-bin range covering ``band`` for a full-length window."""
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    lo = int(np.searchsorted(freqs, band[0], side="left"))
    hi = int(np.searchsorted(freqs, band[1], side="right"))
    return lo, hi


def fft_torch(x, fs: float = 500.0,
              band: tuple[float, float] = FFT_BAND_HZ):
    """Batched one-sided log-magnitude spectrum, still 1-D along frequency.

    ``x``: ``(B, C, T)`` float tensor.  Returns ``(B, C, F)`` containing
    ``log(1 + |rFFT(x)|)`` over the retained band.

    Magnitude only: phase is discarded, so this representation carries global
    spectral content without waveform timing.  That is the intended contrast
    with Raw (full temporal morphology) rather than a limitation to work around.
    """
    import torch

    spec = torch.fft.rfft(x.float(), dim=-1)
    lo, hi = fft_band_bins(x.shape[-1], fs, band)
    return torch.log1p(spec[..., lo:hi].abs())


def stft_band_bins(fs: float = 500.0, n_fft: int = N_FFT,
                   band: tuple[float, float] = STFT_BAND_HZ) -> tuple[int, int]:
    """Inclusive FFT-bin range covering ``band``."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    lo = int(np.searchsorted(freqs, band[0], side="left"))
    hi = int(np.searchsorted(freqs, band[1], side="right"))
    return lo, hi


def stft_torch(x, fs: float = 500.0, n_fft: int = N_FFT, hop: int = HOP,
               band: tuple[float, float] = STFT_BAND_HZ):
    """Batched log-power STFT on GPU.

    ``x``: ``(B, C, T)`` float tensor.  Returns ``(B, C, F, T')`` with
    ``log(1 + |STFT|^2)`` over the retained band.
    """
    import torch

    b, c, t = x.shape
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x.reshape(b * c, t), n_fft=n_fft, hop_length=hop,
                      window=window, center=True, return_complex=True,
                      pad_mode="constant")
    lo, hi = stft_band_bins(fs, n_fft, band)
    power = spec[:, lo:hi, :].abs().pow(2)
    return torch.log1p(power).reshape(b, c, hi - lo, -1)
