"""Reading the four source databases into one harmonised signal cache.

All four are 12-lead, 500 Hz WFDB.  Georgia, Chapman-Shaoxing and Ningbo use the
PhysioNet/CinC ``.mat`` container; PTB-XL 1.0.1 uses plain format-16 ``.dat``.
The cache stores every record as ``float16`` in millivolts, band-limited and
length-standardised, so that the representation and training stages never touch
the raw archives again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STANDARD_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6")
TARGET_FS = 500
TARGET_SAMPLES = 10 * TARGET_FS          # 10 s at 500 Hz

# Mains frequency of the recording site, used for notch filtering.
MAINS_HZ = {"PTBXL": 50.0, "Georgia": 60.0, "Chapman": 50.0, "Ningbo": 50.0}

# --- ADC gain overrides ------------------------------------------------------
# Georgia (G12EC) headers declare 4880 ADC units/mV, while PTB-XL, Chapman and
# Ningbo all declare 1000.  Taking 4880 at face value makes the Georgia signals
# 4.88x too small, which is demonstrably wrong on three independent checks:
#
#   * median worst-limb-lead QRS peak-to-peak is 0.27 mV, against 1.22-1.28 mV
#     for the other three sources -- not physiological for whole recordings;
#   * 94.1% of Georgia records then fall below the clinical low-QRS-voltage
#     threshold (<0.5 mV in all limb leads) while only 3.6% carry the LQRSV
#     label, against 0.4-0.5% sub-threshold in the other sources;
#   * rescaling by exactly 4880/1000 puts Georgia's amplitude distribution on
#     top of the other three (1.32 mV vs 1.22-1.28 mV).
#
# The stored samples are therefore already scaled for a gain of 1000; the
# declared 4880 is the pre-conversion value.  Left uncorrected, the model
# predicted LQRSV on 97.4% of Georgia records.
GAIN_OVERRIDE: dict[str, float] = {"Georgia": 1000.0}

# Plausible range for the median worst-limb-lead QRS peak-to-peak amplitude of a
# source, in mV.  Used by the cache builder to catch a mis-scaled source.
PLAUSIBLE_LIMB_P2P_MV = (0.8, 2.5)


@dataclass
class Header:
    name: str
    n_sig: int
    fs: float
    n_samples: int
    leads: list[str]
    gains: list[float]
    baselines: list[int]
    adc_zeros: list[int]
    fmt: list[str]
    dx: list[str] = field(default_factory=list)
    age: float = float("nan")
    sex: str = ""
    data_file: str = ""


_GAIN_RE = re.compile(r"^([-\d.eE+]+)")


def parse_header(text: str, name: str) -> Header:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    first = lines[0].split()
    n_sig, fs, n_samples = int(first[1]), float(first[2]), int(first[3])

    leads, gains, baselines, adc_zeros, fmts = [], [], [], [], []
    data_file = ""
    for ln in lines[1:1 + n_sig]:
        parts = ln.split()
        data_file = parts[0]
        fmts.append(parts[1])
        gain_field = parts[2]
        gain = float(_GAIN_RE.match(gain_field).group(1))
        # "gain(baseline)/units" -- baseline in parentheses overrides adc_zero
        paren = re.search(r"\(([-\d]+)\)", gain_field)
        baselines.append(int(paren.group(1)) if paren else None)
        adc_zeros.append(int(parts[4]))
        gains.append(gain if gain else 200.0)   # 0 means "unspecified" in WFDB
        leads.append(parts[-1])

    baselines = [b if b is not None else z for b, z in zip(baselines, adc_zeros)]

    hdr = Header(name=name, n_sig=n_sig, fs=fs, n_samples=n_samples, leads=leads,
                 gains=gains, baselines=baselines, adc_zeros=adc_zeros, fmt=fmts,
                 data_file=data_file)

    for ln in lines[n_sig + 1:]:
        if ln.startswith("#Dx:"):
            hdr.dx = [c.strip() for c in ln[4:].split(",") if c.strip()]
        elif ln.startswith("#Age:"):
            try:
                hdr.age = float(ln[5:].strip())
            except ValueError:
                pass
        elif ln.startswith("#Sex:"):
            hdr.sex = ln[5:].strip()
    return hdr


def read_signal_mv(record_path: Path, hdr: Header,
                   source: str | None = None) -> np.ndarray:
    """Return the record as ``(n_sig, n_samples)`` float32 in millivolts.

    ``source`` enables the per-source ADC-gain overrides in ``GAIN_OVERRIDE``.
    """
    mat = record_path.with_suffix(".mat")
    if mat.exists():
        from scipy.io import loadmat
        raw = np.asarray(loadmat(mat)["val"], dtype=np.float32)
    else:
        dat = record_path.with_suffix(".dat")
        flat = np.fromfile(dat, dtype=np.int16)
        raw = flat.reshape(-1, hdr.n_sig).T.astype(np.float32)

    override = GAIN_OVERRIDE.get(source) if source else None
    gains = (np.full(hdr.n_sig, override, dtype=np.float32) if override
             else np.asarray(hdr.gains, dtype=np.float32))[:, None]
    baselines = np.asarray(hdr.baselines, dtype=np.float32)[:, None]
    return (raw - baselines) / gains


def limb_lead_p2p(signal12: np.ndarray) -> float:
    """Largest limb-lead QRS peak-to-peak amplitude, in the signal's units.

    The clinical low-QRS-voltage criterion is stated on the limb leads, so this
    is the quantity to sanity-check a source's amplitude scale against.
    """
    limb = signal12[:6]
    return float((limb.max(axis=1) - limb.min(axis=1)).max())


def reorder_leads(signal: np.ndarray, leads: list[str]) -> np.ndarray:
    """Permute rows into ``STANDARD_LEADS`` order (case-insensitive names)."""
    index = {ln.strip().upper(): i for i, ln in enumerate(leads)}
    try:
        order = [index[ln.upper()] for ln in STANDARD_LEADS]
    except KeyError as exc:
        raise ValueError(f"missing lead {exc} in {leads}") from exc
    return signal[order]


def standardise_length(signal: np.ndarray, n: int = TARGET_SAMPLES) -> np.ndarray:
    """Deterministic 10 s window: keep the leading ``n`` samples, right-pad with
    zeros when the record is shorter.  No random cropping anywhere in the study.
    """
    have = signal.shape[1]
    if have == n:
        return signal
    if have > n:
        return signal[:, :n]
    out = np.zeros((signal.shape[0], n), dtype=signal.dtype)
    out[:, :have] = signal
    return out


def resample_to(signal: np.ndarray, fs_in: float, fs_out: int = TARGET_FS) -> np.ndarray:
    if int(fs_in) == fs_out:
        return signal
    from scipy.signal import resample_poly
    from math import gcd
    a, b = int(round(fs_out)), int(round(fs_in))
    g = gcd(a, b)
    return resample_poly(signal, a // g, b // g, axis=1).astype(np.float32)
