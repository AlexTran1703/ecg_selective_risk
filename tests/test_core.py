r"""Unit tests for the parts of the study whose correctness is not visually obvious.

    .\.venv\Scripts\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecgsr import labels as L                                     # noqa: E402
from ecgsr.ingest import parse_header, standardise_length         # noqa: E402
from ecgsr.ltt import (binomial_lcb, calibrate_threshold,         # noqa: E402
                       hoeffding_bentkus_pvalue, mean_ucb)
from ecgsr.metrics import fit_thresholds                          # noqa: E402
from ecgsr.representations import to_vcg, INVERSE_DOWER           # noqa: E402
from ecgsr.selective import (aurc, coverage_at_risk, jaccard_loss,  # noqa: E402
                             margin_score, predict_sets,
                             risk_coverage_curve, selective_risk,
                             threshold_grid)


# --------------------------------------------------------------------------- #
#  Jaccard loss
# --------------------------------------------------------------------------- #
def test_jaccard_perfect_and_disjoint():
    y = np.array([[1, 0, 1], [0, 1, 0]])
    assert jaccard_loss(y, y).tolist() == [0.0, 0.0]
    assert jaccard_loss(np.array([[1, 0, 0]]), np.array([[0, 1, 0]]))[0] == 1.0


def test_jaccard_partial_match():
    # true {A,B,C}, predicted {A,B}  ->  J = 2/3, L = 1/3
    loss = jaccard_loss(np.array([[1, 1, 1]]), np.array([[1, 1, 0]]))
    assert loss[0] == pytest.approx(1 / 3)


def test_jaccard_both_empty_is_perfect():
    """A record with no retained diagnosis, predicted as such, is correct --
    not an undefined 0/0 and not a maximal loss."""
    assert jaccard_loss(np.zeros((1, 5)), np.zeros((1, 5)))[0] == 0.0


def test_jaccard_empty_truth_with_false_positive():
    assert jaccard_loss(np.zeros((1, 3)), np.array([[1, 0, 0]]))[0] == 1.0


def test_jaccard_is_bounded():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (500, 13))
    p = rng.integers(0, 2, (500, 13))
    loss = jaccard_loss(y, p)
    assert loss.min() >= 0.0 and loss.max() <= 1.0


# --------------------------------------------------------------------------- #
#  Confidence score
# --------------------------------------------------------------------------- #
def test_margin_is_zero_on_the_decision_boundary():
    probs = np.array([[0.5, 0.9]])
    thresholds = np.array([0.5, 0.5])
    assert margin_score(probs, thresholds)[0] == pytest.approx(0.0)


def test_margin_uses_the_worst_class_not_the_best():
    """One ambiguous class must make the whole record unconfident, even if every
    other class is decided with total certainty."""
    probs = np.array([[0.999, 0.001, 0.51]])
    thresholds = np.array([0.5, 0.5, 0.5])
    s = margin_score(probs, thresholds)[0]
    assert s == pytest.approx(abs(np.log(0.51 / 0.49)), abs=1e-6)


def test_margin_respects_per_class_thresholds():
    # p = 0.3 against t = 0.3 is on the boundary, not confidently negative.
    assert margin_score(np.array([[0.3]]), np.array([0.3]))[0] == pytest.approx(0.0)


def test_predict_sets_uses_per_class_thresholds():
    probs = np.array([[0.4, 0.4]])
    out = predict_sets(probs, np.array([0.3, 0.5]))
    assert out.tolist() == [[1, 0]]


# --------------------------------------------------------------------------- #
#  Risk-coverage
# --------------------------------------------------------------------------- #
def test_risk_coverage_is_monotone_for_perfectly_ordered_scores():
    losses = np.linspace(0, 1, 100)          # loss rises as confidence falls
    scores = np.linspace(1, 0, 100)
    rc = risk_coverage_curve(losses, scores)
    assert np.all(np.diff(rc.risk) >= -1e-12)
    assert rc.coverage[-1] == pytest.approx(1.0)
    assert rc.risk[-1] == pytest.approx(losses.mean())


def test_aurc_prefers_a_better_ordering():
    rng = np.random.default_rng(0)
    losses = rng.uniform(size=2000)
    good = -losses                            # confidence anti-correlated with loss
    bad = rng.uniform(size=2000)              # uninformative
    assert aurc(losses, good) < aurc(losses, bad)


def test_selective_risk_at_full_coverage_is_the_mean_loss():
    losses = np.array([0.0, 0.5, 1.0])
    cov, risk = selective_risk(losses, np.array([3.0, 2.0, 1.0]), tau=0.0)
    assert cov == 1.0 and risk == pytest.approx(0.5)


def test_coverage_at_risk_is_zero_when_unreachable():
    losses = np.full(100, 0.9)
    assert coverage_at_risk(losses, np.linspace(0, 1, 100), alpha=0.1) == 0.0


# --------------------------------------------------------------------------- #
#  Bounds
# --------------------------------------------------------------------------- #
def test_ucb_is_above_the_estimate_and_shrinks_with_n():
    wide = mean_ucb(0.1, n=100, delta=0.05)
    tight = mean_ucb(0.1, n=10_000, delta=0.05)
    assert 0.1 < tight < wide <= 1.0


def test_pvalue_is_one_when_estimate_exceeds_the_null():
    assert hoeffding_bentkus_pvalue(0.3, 0.1, 1000) == 1.0


def test_binomial_lcb_is_below_the_estimate():
    assert 0.0 < binomial_lcb(500, 1000, 0.05) < 0.5


def test_binomial_lcb_handles_degenerate_counts():
    assert binomial_lcb(0, 100, 0.05) == 0.0
    assert 0.0 < binomial_lcb(100, 100, 0.05) < 1.0


# --------------------------------------------------------------------------- #
#  Calibration
# --------------------------------------------------------------------------- #
def test_calibration_reports_failure_rather_than_guessing():
    """When no threshold can meet alpha, the procedure must say so instead of
    silently returning an uncertified threshold."""
    rng = np.random.default_rng(0)
    losses = np.full(500, 0.9)
    scores = rng.uniform(size=500)
    cal = calibrate_threshold(losses, scores, threshold_grid(scores, 50),
                              alpha=0.05, delta=0.05)
    assert not cal.controlled and cal.coverage_cal == 0.0


def test_calibration_accepts_everything_when_the_model_is_flawless():
    rng = np.random.default_rng(0)
    losses = np.zeros(5000)
    scores = rng.uniform(size=5000)
    cal = calibrate_threshold(losses, scores, threshold_grid(scores, 50),
                              alpha=0.10, delta=0.05)
    assert cal.controlled and cal.coverage_cal > 0.95


def test_calibration_controls_risk_on_held_out_data():
    """The guarantee itself: over repeated draws, the test risk must exceed
    alpha at most delta of the time."""
    alpha, delta, violations, trials = 0.10, 0.05, 0, 60
    rng = np.random.default_rng(7)
    for _ in range(trials):
        def draw(n):
            s = rng.gamma(2.0, 1.5, size=n)
            bad = 1.0 / (1.0 + np.exp(1.2 * (s - 2.2)))
            return s, rng.beta(2.0, 3.0, size=n) * (bad > rng.uniform(size=n))

        s_cal, l_cal = draw(2000)
        s_test, l_test = draw(4000)
        cal = calibrate_threshold(l_cal, s_cal, threshold_grid(s_cal, 50), alpha, delta)
        _, risk = selective_risk(l_test, s_test, cal.tau)
        violations += risk > alpha
    assert violations / trials <= delta


# --------------------------------------------------------------------------- #
#  Thresholds
# --------------------------------------------------------------------------- #
def test_fit_thresholds_separates_a_clean_class():
    y = np.array([[0]] * 50 + [[1]] * 50)
    probs = np.array([[0.1]] * 50 + [[0.9]] * 50)
    t = fit_thresholds(y, probs)[0]
    assert 0.1 < t <= 0.9


def test_fit_thresholds_ignores_classes_with_no_positives():
    y = np.zeros((20, 2), dtype=int)
    y[:10, 0] = 1
    probs = np.column_stack([np.linspace(0.9, 0.1, 20), np.full(20, 0.4)])
    assert fit_thresholds(y, probs)[1] == 0.5        # untouched default


# --------------------------------------------------------------------------- #
#  Signal handling
# --------------------------------------------------------------------------- #
def test_vcg_matches_the_inverse_dower_definition():
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(12, 200)).astype(np.float32)
    independent = sig[[0, 1, 6, 7, 8, 9, 10, 11]]
    assert np.allclose(to_vcg(sig), INVERSE_DOWER @ independent, atol=1e-5)


def test_vcg_is_linear_in_amplitude():
    """The transform runs on physical millivolts, so doubling the recording must
    double the VCG -- this is what would break if it were applied after
    per-record normalisation."""
    rng = np.random.default_rng(1)
    sig = rng.normal(size=(12, 100)).astype(np.float32)
    assert np.allclose(to_vcg(2 * sig), 2 * to_vcg(sig), atol=1e-5)


def test_standardise_length_crops_and_pads_deterministically():
    long = np.ones((12, 6000), dtype=np.float32)
    assert standardise_length(long, 5000).shape == (12, 5000)
    short = np.ones((12, 2500), dtype=np.float32)
    out = standardise_length(short, 5000)
    assert out.shape == (12, 5000)
    assert out[:, 2500:].sum() == 0.0 and out[:, :2500].sum() == 12 * 2500


def test_parse_header_reads_gain_baseline_and_dx():
    text = ("E1 2 500 5000\n"
            "E1.mat 16+24 1000.0(12)/mV 16 0 5 0 0 I\n"
            "E1.mat 16+24 1000.0/mV 16 0 5 0 0 II\n"
            "#Age: 61\n#Sex: Male\n#Dx: 426783006,164889003\n")
    hdr = parse_header(text, "E1")
    assert hdr.fs == 500 and hdr.n_samples == 5000
    assert hdr.gains == [1000.0, 1000.0]
    assert hdr.baselines == [12, 0]              # parenthesised value wins
    assert hdr.leads == ["I", "II"]
    assert hdr.dx == ["426783006", "164889003"]
    assert hdr.age == 61.0 and hdr.sex == "Male"


# --------------------------------------------------------------------------- #
#  Label harmonisation
# --------------------------------------------------------------------------- #
def test_equivalence_pairs_collapse_to_one_code():
    equiv = L.build_equivalence_map()
    assert equiv["59118001"] == equiv["713427006"]        # RBBB == CRBBB
    assert L.canonicalise({"59118001", "713427006"}, equiv) == {"713427006"}


def test_ptbxl_sinus_rhythm_is_the_union_of_sr_and_norm():
    assert "426783006" in L.ptbxl_record_labels("{'NORM': 100.0}", "")
    assert "426783006" in L.ptbxl_record_labels("{'SR': 0.0}", "")


def test_ptbxl_axis_codes_come_from_the_heart_axis_column():
    assert "39732003" in L.ptbxl_record_labels("{'NORM': 100.0}", "ALAD")
    assert "39732003" not in L.ptbxl_record_labels("{'NORM': 100.0}", "MID")


def test_ptbxl_pvc_is_not_mapped_to_the_scored_pvc_class():
    """The Challenge routes PTB-XL's PVC statement to the unscored VEB code, and
    reports zero scored PVC in PTB-XL. Mapping it to 427172004 would silently
    admit a class that the pre-registered rule should drop."""
    assert "427172004" not in L.ptbxl_record_labels("{'PVC': 100.0}", "")


def test_common_class_rule_needs_every_source():
    label_sets = {
        "A": [{"1"}] * 150 + [{"2"}] * 150,
        "B": [{"1"}] * 150 + [{"2"}] * 50,      # class 2 too rare in B
    }
    keep, counts = L.select_common_classes(label_sets, {"1", "2"}, min_per_source=100)
    assert keep == ["1"]
    assert counts["2"] == {"A": 150, "B": 50}


def test_common_class_rule_counts_merged_pairs_together():
    label_sets = {
        "A": [{"59118001"}] * 60 + [{"713427006"}] * 60,
        "B": [{"713427006"}] * 120,
    }
    keep, _ = L.select_common_classes(label_sets, {"59118001", "713427006"},
                                      min_per_source=100)
    assert keep == ["713427006"]        # 120 in each source once merged


# --------------------------------------------------------------------------- #
#  Splits
# --------------------------------------------------------------------------- #
def test_split_seed_is_stable_across_processes():
    """Every pipeline for a given held-out source must get the *same* split,
    otherwise the Raw/VCG/STFT comparison is not controlled. Python salts str
    hashing per process, so the seed must not come from hash()."""
    from ecgsr.datasets import _stable_seed
    assert _stable_seed("PTBXL", 0) == 3293061994
    assert _stable_seed("PTBXL", 0) != _stable_seed("Georgia", 0)
    assert _stable_seed("PTBXL", 0) != _stable_seed("PTBXL", 1)


# --------------------------------------------------------------------------- #
#  Numerical robustness
# --------------------------------------------------------------------------- #
def test_metrics_survive_nan_predictions():
    """A NaN from the model must degrade the score, not raise -- a metric call
    should never be able to kill a multi-hour training run."""
    from ecgsr.metrics import macro_auprc, macro_auroc, per_class_auroc
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (200, 4))
    probs = rng.uniform(size=(200, 4))
    probs[5, 1] = np.nan
    probs[7, 2] = np.inf
    assert np.isfinite(macro_auroc(y, probs))
    assert np.isfinite(macro_auprc(y, probs))
    assert np.isfinite(per_class_auroc(y, probs)).all()


def test_fit_thresholds_survives_nan_predictions():
    from ecgsr.metrics import fit_thresholds
    y = np.array([[0]] * 50 + [[1]] * 50)
    probs = np.array([[0.1]] * 50 + [[0.9]] * 50)
    probs[3, 0] = np.nan
    assert np.isfinite(fit_thresholds(y, probs)).all()


def test_bfloat16_is_the_default_autocast_dtype_on_cuda():
    """float16 autocast overflowed this backbone late in training; bf16 has
    fp32's exponent range and is the intended default wherever it is available."""
    import torch
    from ecgsr.train import TrainConfig, amp_dtype_for
    cfg = TrainConfig()
    assert cfg.amp_dtype == "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        assert amp_dtype_for(cfg, torch.device("cuda")) is torch.bfloat16


# --------------------------------------------------------------------------- #
#  Amplitude scale
# --------------------------------------------------------------------------- #
def test_georgia_gain_override_is_applied():
    """Georgia headers declare 4880 ADC units/mV but the samples are stored at
    1000, which made every Georgia signal 4.88x too small and drove the model to
    predict low-QRS-voltage on 97% of that source."""
    from ecgsr.ingest import GAIN_OVERRIDE
    assert GAIN_OVERRIDE["Georgia"] == 1000.0
    for other in ("PTBXL", "Chapman", "Ningbo"):
        assert other not in GAIN_OVERRIDE


def test_read_signal_mv_honours_the_override(tmp_path):
    import numpy as np
    from scipy.io import savemat
    from ecgsr.ingest import parse_header, read_signal_mv

    text = ("R 2 500 4\n"
            "R.mat 16+24 4880/mV 16 0 0 0 0 I\n"
            "R.mat 16+24 4880/mV 16 0 0 0 0 II\n")
    hdr = parse_header(text, "R")
    savemat(tmp_path / "R.mat", {"val": np.full((2, 4), 4880, dtype=np.int16)})

    declared = read_signal_mv(tmp_path / "R", hdr, source="Chapman")
    overridden = read_signal_mv(tmp_path / "R", hdr, source="Georgia")
    assert np.allclose(declared, 1.0)              # 4880 / 4880
    assert np.allclose(overridden, 4.88)           # 4880 / 1000
    # No source given => declared gain, so existing behaviour is unchanged.
    assert np.allclose(read_signal_mv(tmp_path / "R", hdr), 1.0)


def test_limb_lead_p2p_uses_worst_limb_lead():
    import numpy as np
    from ecgsr.ingest import limb_lead_p2p
    sig = np.zeros((12, 100), dtype=np.float32)
    sig[1, :50] = 1.5                              # lead II swing of 1.5 mV
    sig[7, :50] = 9.0                              # V2: must be ignored
    assert limb_lead_p2p(sig) == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
#  FFT representation
# --------------------------------------------------------------------------- #
def test_fft_band_covers_exactly_the_preprocessing_band():
    """10 s at 500 Hz gives df = 0.1 Hz, so 0.5-50 Hz is 496 bins per lead."""
    import numpy as np
    from ecgsr.representations import FFT_BAND_HZ, fft_band_bins
    lo, hi = fft_band_bins(5000, 500.0, FFT_BAND_HZ)
    freqs = np.fft.rfftfreq(5000, 1 / 500.0)
    assert hi - lo == 496
    assert freqs[lo] == pytest.approx(0.5)
    assert freqs[hi - 1] == pytest.approx(50.0)


def test_fft_is_one_dimensional_and_keeps_lead_channels():
    import torch
    from ecgsr.representations import fft_torch
    z = fft_torch(torch.randn(4, 12, 5000))
    assert z.shape == (4, 12, 496)         # still 1-D along frequency
    assert torch.isfinite(z).all()
    assert (z >= 0).all()                  # log1p of a magnitude


def test_fft_is_shift_invariant_but_raw_is_not():
    """Magnitude discards phase, which is the intended Raw/FFT contrast: a
    circular time shift leaves the spectrum unchanged but not the waveform."""
    import torch
    from ecgsr.representations import fft_torch
    x = torch.randn(2, 12, 5000)
    rolled = torch.roll(x, 137, dims=-1)
    assert torch.allclose(fft_torch(x), fft_torch(rolled), atol=1e-3)
    assert not torch.allclose(x, rolled)


def test_all_study_representations_share_one_backbone():
    """Raw, VCG and FFT must differ only in input channels -- that is what makes
    this a representation benchmark rather than an architecture benchmark."""
    from ecgsr.datasets import REPRESENTATIONS
    from ecgsr.models import build_model
    assert REPRESENTATIONS == ("raw", "vcg", "fft")
    kinds = {r: type(build_model(r, 13)).__name__ for r in REPRESENTATIONS}
    assert set(kinds.values()) == {"ResNet1d"}, kinds
    n_in = {r: build_model(r, 13).stem[0][0].in_channels for r in REPRESENTATIONS}
    assert n_in == {"raw": 12, "vcg": 3, "fft": 12}


def test_fft_statistics_are_per_frequency_bin():
    """Spectral magnitude falls off steeply with frequency, so a single scalar
    per lead would let the lowest bins dominate."""
    from ecgsr.datasets import Representation
    assert Representation("fft").stat_dims == (0,)
    assert Representation("raw").stat_dims == (0, 2)


# --------------------------------------------------------------------------- #
#  Conditional vs linear risk test
# --------------------------------------------------------------------------- #
def test_conditional_pvalue_is_the_default_and_is_tighter():
    """The linear form's statistic is alpha - C*(alpha-R), so its signal is the
    *product* of coverage and risk margin and vanishes near feasibility. The
    conditional test bounds E[L | s>=tau] directly on the accepted records and
    certifies thresholds the linear form cannot."""
    from ecgsr.ltt import DEFAULT_PVALUE_MODE, selective_risk_pvalue
    assert DEFAULT_PVALUE_MODE == "conditional"

    rng = np.random.default_rng(0)
    n = 2000
    scores = rng.uniform(size=n)
    # Losses are small for confident records, large for the rest.
    losses = np.where(scores > 0.7, rng.beta(1, 24, size=n), rng.beta(4, 6, size=n))
    tau = 0.7
    p_cond = selective_risk_pvalue(losses, scores, tau, 0.10, "conditional")
    p_lin = selective_risk_pvalue(losses, scores, tau, 0.10, "linear")
    assert p_cond < p_lin


def test_pvalue_modes_both_reject_only_when_risk_is_low():
    from ecgsr.ltt import selective_risk_pvalue
    scores = np.linspace(0, 1, 1000)
    bad = np.full(1000, 0.8)
    for mode in ("conditional", "linear"):
        assert selective_risk_pvalue(bad, scores, 0.5, 0.10, mode) == 1.0


def test_unknown_pvalue_mode_is_rejected():
    from ecgsr.ltt import selective_risk_pvalue
    with pytest.raises(ValueError):
        selective_risk_pvalue(np.zeros(10), np.zeros(10), 0.0, 0.1, "nonsense")


def test_conditional_calibration_still_controls_risk():
    """Re-run the guarantee check with the conditional test: over repeated
    draws the test risk must exceed alpha at most delta of the time."""
    from ecgsr.ltt import calibrate_threshold
    from ecgsr.selective import selective_risk, threshold_grid
    alpha, delta, violations, trials = 0.10, 0.05, 0, 80
    rng = np.random.default_rng(11)
    for _ in range(trials):
        def draw(n):
            s = rng.gamma(2.0, 1.5, size=n)
            bad = 1.0 / (1.0 + np.exp(1.2 * (s - 2.2)))
            return s, rng.beta(2.0, 3.0, size=n) * (bad > rng.uniform(size=n))

        s_grid, _ = draw(3000)                 # grid from independent data
        s_cal, l_cal = draw(2000)
        s_test, l_test = draw(4000)
        cal = calibrate_threshold(l_cal, s_cal, threshold_grid(s_grid, 50),
                                  alpha, delta)
        _, risk = selective_risk(l_test, s_test, cal.tau)
        violations += risk > alpha
    assert violations / trials <= delta
