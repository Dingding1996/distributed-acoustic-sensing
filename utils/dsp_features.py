"""
Valencia-IslaLink DAS Dataset - Sliding-Window DSP Feature Extraction
=======================================================================
Single-channel sliding-window feature extraction for DAS anomaly/vessel
detection: time-domain (RMS, Peak, Crest Factor, Kurtosis), spectral
peakiness (Welch PSD max/mean ratio), non-uniform band-energy (Welch PSD),
and optional Wavelet Packet Decomposition sub-band energy.

Feature extraction only -- raw-to-model-ready preprocessing (bad-channel
handling, detrend + bandpass filtering, distance-baseline normalization)
lives in utils/data_preprocess.py. extract_window_features() below expects
an already-preprocessed signal; it does not filter.

This is a clean replacement of the predecessor bearing-fault-diagnosis
version of this module (time_domain_features, frequency_domain_features,
envelope_analysis/envelope_features for BPFO/BPFI, etc.) -- none of that
applies to DAS strain-rate data, see CLAUDE.md for the full rationale.
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis
from typing import List, Tuple

# Try to import pywt; WPD features are skipped (not approximated) if unavailable.
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def extract_window_features(
    sig: np.ndarray,
    fs: float,
    window_sec: float,
    hop_sec: float,
    psd_bands: List[Tuple[float, float]],
    wpd_wavelet: str = "db4",
    wpd_level: int = 3,
    use_wpd: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Sliding-window DSP feature extraction for one DAS channel signal.

    Computes per-window RMS, Peak, Crest Factor, Kurtosis, spectral
    peakiness (Welch PSD max/mean ratio -- how "tonal" vs broadband the
    window is, since a vessel's narrow tonal peak is exactly what coarse
    band-energy features alone could miss), per-band PSD energy, and
    optional Wavelet Packet Decomposition sub-band energy. One window = one
    row of the returned feature matrix. Called by the notebook's batch
    extraction (Section 3a).

    Args:
        sig: 1D filtered strain-rate signal for one channel (already
            detrended + bandpassed upstream -- this function does not filter).
        fs: Sampling rate, Hz.
        window_sec: Sliding-window length, seconds.
        hop_sec: Sliding-window hop, seconds.
        psd_bands: List of (f_low, f_high) Hz tuples for band-energy features.
        wpd_wavelet: PyWavelets wavelet name (only used if use_wpd).
        wpd_level: WPD decomposition level (only used if use_wpd) -- produces
            2**wpd_level equal-width sub-bands spanning 0-Nyquist.
        use_wpd: Whether to compute WPD sub-band energy (requires HAS_PYWT;
            silently skipped if pywt isn't installed, same as the rest of
            this project's optional-dependency convention).

    Returns:
        (feats, feature_names): feats is (n_windows, n_features) float64;
        feature_names is the matching list of column names, in the order
        rms, peak, crest, kurt, peakiness, band_<lo>-<hi>hz (one per
        psd_bands entry), then wpd_<lo>-<hi>hz (one per WPD sub-band, only
        if use_wpd).

    Raises:
        ValueError: If sig is shorter than one window.
    """
    win_samp = int(window_sec * fs)
    hop_samp = int(hop_sec * fs)
    if len(sig) < win_samp:
        raise ValueError(f"signal length {len(sig)} shorter than one window ({win_samp} samples)")

    starts = list(range(0, len(sig) - win_samp + 1, hop_samp))
    n_windows = len(starts)
    n_bands = len(psd_bands)
    use_wpd = use_wpd and HAS_PYWT

    rms_v = np.zeros(n_windows)
    peak_v = np.zeros(n_windows)
    crest_v = np.zeros(n_windows)
    kurt_v = np.zeros(n_windows)
    peakiness_v = np.zeros(n_windows)
    band_v = np.zeros((n_windows, n_bands))
    if use_wpd:
        wpd_n = 2 ** wpd_level
        wpd_v = np.zeros((n_windows, wpd_n))

    for wi, s0 in enumerate(starts):
        seg = sig[s0:s0 + win_samp].astype(np.float64)
        rms = np.sqrt(np.mean(seg ** 2))
        rms_v[wi] = rms
        peak_v[wi] = np.max(np.abs(seg))
        crest_v[wi] = peak_v[wi] / (rms + 1e-30)
        kurt_v[wi] = kurtosis(seg)

        f_p, pxx = welch(seg, fs=fs, nperseg=min(len(seg), 256))
        peakiness_v[wi] = pxx.max() / (pxx.mean() + 1e-30)
        for bi, (lo, hi) in enumerate(psd_bands):
            m = (f_p >= lo) & (f_p < hi)
            band_v[wi, bi] = pxx[m].mean() if m.any() else 0.0

        if use_wpd:
            wp = pywt.WaveletPacket(seg, wavelet=wpd_wavelet, mode='symmetric', maxlevel=wpd_level)
            for ni, node in enumerate(wp.get_level(wpd_level, 'freq')):
                wpd_v[wi, ni] = np.sum(node.data ** 2)

    feature_names = ['rms', 'peak', 'crest', 'kurt', 'peakiness'] + [
        f'band_{lo:.0f}-{hi:.0f}hz' for lo, hi in psd_bands
    ]
    feature_cols = [rms_v, peak_v, crest_v, kurt_v, peakiness_v] + [
        band_v[:, bi] for bi in range(n_bands)
    ]
    if use_wpd:
        wpd_bw = (fs / 2) / wpd_n
        feature_names += [
            f'wpd_{int(i * wpd_bw)}-{int((i + 1) * wpd_bw)}hz' for i in range(wpd_n)
        ]
        feature_cols += [wpd_v[:, ni] for ni in range(wpd_n)]

    feats = np.column_stack(feature_cols)
    return feats, feature_names


# ============================================================
# Self-check (shared by notebook pre-flight gate + unit tests)
# ============================================================

def self_check() -> Tuple[np.ndarray, List[str]]:
    """Run extract_window_features() on a synthetic strain-rate-like signal.

    Generates a synthetic signal (a low-frequency tone + broadband noise,
    shaped like a filtered DAS channel) and extracts features over a few
    short windows, then asserts the output is non-empty and finite. Used as
    a fast pre-flight gate before any real data loading (notebook Section 0)
    so a broken DSP pipeline fails in seconds, not after a long extraction.

    Returns:
        (feats, feature_names) from extract_window_features().

    Raises:
        AssertionError: If any sanity check fails.
    """
    rng = np.random.default_rng(42)
    fs, n_samples = 250.0, 5_000
    t = np.arange(n_samples, dtype=np.float64) / fs
    sig = np.sin(2 * np.pi * 15 * t) + 0.3 * rng.standard_normal(n_samples)

    psd_bands = [(2, 20), (20, 50), (50, 124)]
    feats, feature_names = extract_window_features(
        sig, fs, window_sec=1.0, hop_sec=0.5, psd_bands=psd_bands,
        wpd_wavelet='db4', wpd_level=2, use_wpd=HAS_PYWT,
    )

    assert feats.shape[0] > 0, "No windows extracted"
    assert feats.shape[1] == len(feature_names), "Feature matrix/name count mismatch"
    assert np.all(np.isfinite(feats)), "Non-finite feature value(s)"
    assert any(n.startswith("wpd_") for n in feature_names) == HAS_PYWT, \
        "WPD features present/absent inconsistent with HAS_PYWT"

    return feats, feature_names


# ============================================================
# Quick test
# ============================================================
if __name__ == '__main__':
    feats, names = self_check()
    print(f"self_check() OK: {feats.shape[0]} windows x {feats.shape[1]} features")
    print(f"Feature names: {names}")
