"""
Valencia-IslaLink DAS Dataset - Raw-to-Model-Ready Preprocessing Pipeline
=======================================================================
Turns a raw strain-rate array (one or many channels, straight out of
data_loader.load_h5_file()/load_event()) into the filtered,
distance-baseline-normalized signal that utils/dsp_features.py's
extract_window_features() expects: bad-channel screen -> (interpolate or
leave for the caller to skip) -> common-mode removal -> linear-detrend +
Butterworth bandpass -> f-k domain apparent-velocity highpass -> divide by a
fitted distance-baseline RMS curve.

Split out from dsp_features.py so preprocessing (this file) and feature
extraction (dsp_features.py) are two separate, independently testable
concerns -- both are still called directly from the notebook (every EDA
cell and the Section 3a-ii/7b multi-channel batch paths all call
preprocess_channels() here), so the "1 hour" (single representative event)
and "4 hour" (full batch across all downloaded events) code paths can never
quietly drift apart. Because preprocess_channels() is this one shared
pipeline, common-mode removal and f-k filtering apply everywhere it's
called -- including Section 3a-ii's batch feature extraction, so they
affect what actually gets fed to the anomaly-detection models, not just
what Section 2's EDA cells plot.

Also provides stack_local_channels() -- a standalone, NOT-called-by-default
local coherent channel stack (zero-moveout, not beamforming -- see its
docstring), kept separate because it changes the signal's spatial
resolution/meaning in a way that shouldn't silently become part of the
default pipeline the way common-mode removal/fk_filter() now are.
fk_filter() itself is wave-type separation/filtering only (an
apparent-velocity highpass in the f-k domain) -- it does not estimate
source position, bearing, or the apparent velocity of any specific
arrival, which remains out of scope per CLAUDE.md SS1
(beamforming/hyperbola-fitting for source localization).
"""

import numpy as np
from scipy import fft as sp_fft
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, detrend, filtfilt
from typing import Optional, Tuple, Union


def detect_bad_channels(
    strain_rate: np.ndarray,
    low_ratio: float = 0.1,
    high_ratio: float = 5.0,
) -> np.ndarray:
    """Flag channels whose RMS is NaN or an outlier relative to the group median.

    Args:
        strain_rate: (n_samples, n_channels) raw strain-rate array (or
            (n_samples,) for a single channel).
        low_ratio: flag a channel if its RMS is below low_ratio * median RMS.
        high_ratio: flag a channel if its RMS is above high_ratio * median RMS.

    Returns:
        Boolean array of shape (n_channels,) (length 1 for a 1D input), True
        where the channel is flagged.
    """
    rms_ch = np.atleast_1d(np.sqrt(np.nanmean(strain_rate ** 2, axis=0)))
    med_rms = np.nanmedian(rms_ch)
    return np.isnan(rms_ch) | (rms_ch > high_ratio * med_rms) | (rms_ch < low_ratio * med_rms)


def interpolate_bad_channels(strain_rate: np.ndarray, is_bad: np.ndarray) -> np.ndarray:
    """Replace flagged channels with the average of their nearest good neighbours.

    Only meaningful when adjacent array columns are true physical neighbours
    (e.g. a contiguous channel range) -- for a strided/non-contiguous channel
    selection, flag and skip bad channels downstream instead (see
    preprocess_channels' bad_channel_action="skip").

    Args:
        strain_rate: (n_samples, n_channels) array.
        is_bad: (n_channels,) boolean mask from detect_bad_channels().

    Returns:
        Copy of strain_rate with flagged channels interpolated.
    """
    clean = strain_rate.copy()
    for idx in np.where(is_bad)[0]:
        l, r = idx - 1, idx + 1
        while l >= 0 and is_bad[l]:
            l -= 1
        while r < len(is_bad) and is_bad[r]:
            r += 1
        if l >= 0 and r < len(is_bad):
            clean[:, idx] = 0.5 * (strain_rate[:, l] + strain_rate[:, r])
        elif l >= 0:
            clean[:, idx] = strain_rate[:, l]
        else:
            clean[:, idx] = strain_rate[:, r]
    return clean


def remove_common_mode(strain_rate: np.ndarray, method: str = "median") -> np.ndarray:
    """Subtract the per-time-sample disturbance shared across channels.

    Standard DAS denoising technique: at each time sample, whatever
    disturbance shows up on most/all channels simultaneously is treated as
    shared "common-mode" noise (e.g. instrument/laser-source noise, common
    electronic pickup) and removed, leaving each channel's own deviation
    from that shared baseline. This is a per-time-sample operation across
    the channel axis -- unrelated to detrend_bandpass()'s per-channel
    operation across the time axis.

    Not part of preprocess_channels()'s default pipeline -- call this
    explicitly (typically before detrend_bandpass(), since the common-mode
    component can be broadband) when a notebook diagnostic wants it.

    Args:
        strain_rate: (n_samples, n_channels) array. Not meaningful for a
            single channel (n_channels must be >= 2 for "common" to mean
            anything).
        method: "median" (default -- robust to a minority of channels
            carrying a real localized signal at a given instant, which a
            plain mean would partially subtract out too) or "mean".

    Returns:
        (n_samples, n_channels) array with the per-sample common-mode
        component subtracted.

    Raises:
        ValueError: If method is not "median" or "mean", or strain_rate is
            not 2D.
    """
    if strain_rate.ndim != 2:
        raise ValueError(f"remove_common_mode() expects a 2D (n_samples, n_channels) array, got ndim={strain_rate.ndim}")
    if method == "median":
        common = np.nanmedian(strain_rate, axis=1, keepdims=True)
    elif method == "mean":
        common = np.nanmean(strain_rate, axis=1, keepdims=True)
    else:
        raise ValueError(f"method must be 'median' or 'mean', got {method!r}")
    return strain_rate - common


def detrend_bandpass(
    strain_rate: np.ndarray,
    fs: float,
    bandpass: Tuple[float, float],
    order: int = 4,
    max_channels_per_chunk: int = 500,
) -> np.ndarray:
    """Linear-detrend then zero-phase Butterworth bandpass filter.

    Identical logic for a single channel (1D array) or many at once (2D
    array, filtered along axis 0) -- the same call is used for the
    notebook's single-channel demo and its multi-channel batch/EDA paths.

    Args:
        strain_rate: (n_samples,) or (n_samples, n_channels) array.
        fs: Sampling rate, Hz.
        bandpass: (low, high) cutoff frequencies, Hz.
        order: Butterworth filter order.
        max_channels_per_chunk: For 2D input wider than this many channels,
            filter in channel-chunks of this size to bound peak memory --
            detrend/filtfilt operate independently per channel, so chunking
            the channel axis never changes the result. Verified directly:
            a full-deep-water-range single-file array (e.g. (150K, 1959))
            raised a MemoryError inside filtfilt in a long-running notebook
            session (other cells' arrays already resident) even though the
            same call succeeded in an isolated process -- chunking bounds
            peak memory regardless of what else is already in memory.
            Ignored for 1D input.

    Returns:
        Detrended + filtered array, same shape as strain_rate, same dtype as
        strain_rate (the Butterworth coefficients are cast to strain_rate's
        dtype before filtering -- scipy.signal.butter() always returns
        float64, and filtfilt() silently upcasts a float32 signal to match,
        doubling memory for no accuracy benefit this project needs; verified
        directly that float32 coefficients change the filtered output by a
        small fraction of a percent, immaterial next to this pipeline's much
        larger sources of uncertainty -- CLAUDE.md SS9's no-ground-truth
        framing).
    """
    detrended = detrend(strain_rate, axis=0, type='linear')
    b, a = butter(order, bandpass, btype='band', fs=fs)
    b = b.astype(detrended.dtype, copy=False)
    a = a.astype(detrended.dtype, copy=False)

    if detrended.ndim == 2 and detrended.shape[1] > max_channels_per_chunk:
        filtered = np.empty_like(detrended)
        for start in range(0, detrended.shape[1], max_channels_per_chunk):
            end = start + max_channels_per_chunk
            filtered[:, start:end] = filtfilt(b, a, detrended[:, start:end], axis=0)
        return filtered

    return filtfilt(b, a, detrended, axis=0)


def normalize_by_distance(
    signal: np.ndarray,
    distance_km: Union[float, np.ndarray],
    baseline_curve: np.poly1d,
) -> np.ndarray:
    """Divide by a fitted distance-baseline RMS curve (CLAUDE.md SS9).

    Args:
        signal: (n_samples,) single-channel or (n_samples, n_channels) array.
        distance_km: scalar distance (single channel) or a per-channel
            distance array matching signal's second axis.
        baseline_curve: fitted curve (e.g. an np.poly1d from np.polyfit)
            mapping distance_km -> expected baseline RMS.

    Returns:
        signal divided by the baseline curve evaluated at distance_km.
    """
    baseline = baseline_curve(np.asarray(distance_km, dtype=np.float64))
    if signal.ndim == 2:
        baseline = baseline[np.newaxis, :]
    return signal / baseline


def _cosine_taper(x: np.ndarray) -> np.ndarray:
    """Cosine ramp: 0 at x<=0, 1 at x>=1, smooth cosine transition between.

    Ported from old/DAS_Signal_Processing.ipynb's identically-named helper --
    used to avoid Gibbs ringing from a sharp boundary (a time-domain edge
    taper before an FFT, or an f-k domain mask edge).
    """
    return 0.5 * (1 - np.cos(np.pi * np.clip(x, 0, 1)))


def fk_filter(
    signal: np.ndarray,
    fs: float,
    channel_spacing: float,
    c_low: float,
    c_high: Optional[float] = None,
    taper_frac: float = 0.2,
    time_taper_sec: float = 2.0,
) -> np.ndarray:
    """Frequency-wavenumber (f-k) domain apparent-velocity filter.

    Wave-TYPE separation/filtering only -- does NOT estimate the source
    position, bearing, or apparent velocity of any specific arrival (that
    would be beamforming or hyperbola/TDOA fitting, which remain out of
    scope per CLAUDE.md SS1). This function passes only the energy whose
    apparent velocity |f/k| is >= c_low (highpass, c_high=None, the default)
    or in [c_low, c_high] (bandpass, c_high given) -- cosine-tapered edges,
    zeroing out everything else in the f-k domain before transforming back
    to the time-space domain. Same masking math as
    old/DAS_Signal_Processing.ipynb's fk_masks(), generalized to one
    caller-chosen band (or highpass) instead of two hardcoded bands
    (acoustic/Scholte).

    IMPORTANT apparent-velocity vs. true-velocity distinction: |f/k| is the
    velocity measured ALONG THE CABLE, not the wave's true propagation
    speed. For a wave truly propagating at speed c, apparent velocity is
    always >= c regardless of incidence angle -- equal to c only for an
    arrival exactly along the cable axis (end-fire), and unbounded as the
    incidence angle approaches broadside (a near-field source directly
    overhead arrives at every channel almost simultaneously, i.e. apparent
    velocity -> infinity near the crossing point). This is exactly why
    old/DAS_Signal_Processing.ipynb's fk_masks() mask_ac is a HIGHPASS
    (c_app >= c_water), not a band: a real near-field crossing (e.g. a
    vessel passing over the cable) spans apparent velocities from ~c (far
    from the crossing point) up to near-infinite (close to it) -- capping
    c_high would cut out exactly the near-field portion, often the
    strongest part of the signal. Only pass a finite c_high when you
    specifically want to exclude a faster wave type too (e.g. to isolate a
    slower interface/Scholte wave from a faster acoustic one, as
    old/DAS_Signal_Processing.ipynb's mask_sc does).

    Args:
        signal: (n_samples, n_channels) array, time along axis 0, channel
            along axis 1. Not meaningful for a single channel (a 2D f-k
            transform needs a real spatial axis).
        fs: Sampling rate, Hz.
        channel_spacing: Distance between adjacent channels, metres (must
            match the actual spacing of signal's columns -- e.g. multiply by
            CHANNEL_STRIDE if signal was spatially subsampled).
        c_low: Lower apparent-velocity bound, m/s -- physically motivated by
            a hypothesised TRUE propagation speed (see note above), not an
            arbitrary threshold.
        c_high: Upper apparent-velocity bound, m/s, or None (default) for a
            highpass with no upper bound -- see note above for why this is
            the right default for a near-field arrival.
        taper_frac: Fractional cosine-taper width around c_low/c_high (0.2
            matches the archived notebook's default).
        time_taper_sec: ABSOLUTE duration (seconds) tapered at each time edge
            before the 2D FFT -- avoids a time-edge discontinuity leaking
            across the whole f-k plane. Deliberately a fixed duration, not a
            fraction of the record length (old/DAS_Signal_Processing.ipynb's
            own `_tlen = int(0.1 * n_time)` convention): verified directly
            that a fractional taper corrupts the FIRST ~60s of any 10-minute
            (or longer) record it's applied to when time_taper_frac=0.1 --
            for callers that then display/use only the first N seconds of
            that record (e.g. this notebook's 60s EDA display window), the
            ENTIRE displayed region fell inside the taper's ramp-up zone,
            visibly (and wrongly) suppressing amplitude near t=0 that has
            nothing to do with the actual signal. 2.0s is a fixed, small
            edge effect regardless of how long the input record is.

    Returns:
        (n_samples, n_channels) real-valued array: signal with only the
        passed apparent-velocity range (highpass or band) retained.

    Raises:
        ValueError: If signal is not 2D, or c_high is given and <= c_low.
    """
    if signal.ndim != 2:
        raise ValueError(f"fk_filter() expects a 2D (n_samples, n_channels) array, got ndim={signal.ndim}")
    if c_high is not None and c_high <= c_low:
        raise ValueError(f"c_high ({c_high}) must be greater than c_low ({c_low})")

    n_samples, n_channels = signal.shape
    real_dtype = signal.dtype  # preserve caller's precision -- see below

    # taper/mask are built in real_dtype (not a hardcoded float64), AND the
    # FFT calls below use scipy.fft (not numpy.fft) -- both are needed for the
    # 2D FFT to actually stay complex64 for float32 input. numpy.fft.fft2()
    # unconditionally computes in complex128 regardless of input dtype
    # (verified directly: a float32 array in still comes back complex128) --
    # scipy.fft.fft2() is the one that actually preserves single precision.
    # Getting only the taper/mask dtype right (without also switching to
    # scipy.fft) would look like a fix but silently upcast right back to
    # complex128 inside fft2() itself. This is the same class of bug fixed
    # earlier in detrend_bandpass() (butter() coefficients forced float64),
    # except here it silently doubles a 2D COMPLEX array instead of a 1D real
    # one. Verified directly: this was the actual cause of a MemoryError
    # allocating a (1959, 150250) complex128 array (4.39 GiB) for one 10-minute,
    # full-deep-water-channel-range file -- complex64 halves every array below.
    tlen = max(1, min(int(time_taper_sec * fs), n_samples // 2))
    taper = np.ones(n_samples, dtype=real_dtype)
    taper[:tlen] = _cosine_taper(np.arange(tlen) / tlen).astype(real_dtype)
    taper[-tlen:] = _cosine_taper(np.arange(tlen, 0, -1) / tlen).astype(real_dtype)
    tapered = np.nan_to_num(signal * taper[:, np.newaxis], nan=0.0)

    fk = sp_fft.fftshift(sp_fft.fft2(tapered))
    freqs = sp_fft.fftshift(sp_fft.fftfreq(n_samples, d=1 / fs)).astype(real_dtype)
    wavenumbers = sp_fft.fftshift(sp_fft.fftfreq(n_channels, d=channel_spacing)).astype(real_dtype)
    # Broadcast (n_samples, 1) x (1, n_channels) instead of np.meshgrid()'s
    # full (n_samples, n_channels) ff/kk arrays -- avoids materializing 2 extra
    # full-size arrays (plus their abs() copies) before c_app/mask, which even
    # in real_dtype are the same size as fk itself.
    abs_f = np.abs(freqs)[:, np.newaxis]
    abs_k = np.abs(wavenumbers)[np.newaxis, :]

    with np.errstate(divide='ignore', invalid='ignore'):
        c_app = np.where(abs_k > 0, abs_f / abs_k, np.inf).astype(real_dtype)
    lo_lo, lo_hi = c_low * (1 - taper_frac), c_low * (1 + taper_frac)
    mask = _cosine_taper((c_app - lo_lo) / (lo_hi - lo_lo)).astype(real_dtype)
    if c_high is not None:
        hi_lo, hi_hi = c_high * (1 - taper_frac), c_high * (1 + taper_frac)
        mask *= (1 - _cosine_taper((c_app - hi_lo) / (hi_hi - hi_lo))).astype(real_dtype)

    # In-place fk *= mask (not fk * mask) -- fk is never used again, so this
    # avoids allocating a separate same-size temporary just to hold the product.
    fk *= mask
    filtered = np.real(sp_fft.ifft2(sp_fft.ifftshift(fk)))
    return filtered.astype(signal.dtype)


def stack_local_channels(
    signal: np.ndarray,
    n_stack: int,
    mode: str = "mean",
) -> np.ndarray:
    """Naive (zero-moveout) local coherent stack across adjacent channels.

    This is NOT beamforming or source localization -- it applies no time
    shift, assumes no apparent velocity, and estimates no source position or
    bearing (that work remains out of scope per CLAUDE.md SS1). It simply
    sums/averages each channel with its n_stack-1 nearest neighbours (a
    sliding window along the CHANNEL axis, not time) to reduce
    per-channel-independent noise -- valid only when the true moveout
    (arrival-time difference) across the n_stack-channel span is small
    relative to one sample period. Concretely: at fs=250 Hz one sample
    period is 4 ms, so for a hypothesised apparent velocity c_apparent, this
    naive stack is only valid while
    (n_stack * channel_spacing) / c_apparent << 1 / fs -- check that bound
    for your chosen n_stack and dataset before trusting the result. A
    velocity-corrected (delay-and-sum) variant would extend this into
    beamforming territory and is deliberately not implemented here.

    Not part of preprocess_channels()'s default pipeline -- call this
    explicitly; it changes the spatial resolution of the signal (adjacent
    channels become correlated by construction), so it must never silently
    become the default modelling input.

    Args:
        signal: (n_samples, n_channels) already-preprocessed signal (same
            shape preprocess_channels() returns) -- not meant to be called
            on raw strain rate.
        n_stack: Number of adjacent channels to combine. Odd values centre
            the window on each channel; even values are biased slightly
            toward one side (scipy.ndimage.uniform_filter1d's convention) --
            not silently rounded, so pick deliberately.
        mode: "mean" (default -- preserves signal amplitude, comparable
            across different n_stack choices) or "sum" (raw sum -- amplitude
            scales with n_stack, only meaningful for relative/visual
            comparison).

    Returns:
        (n_samples, n_channels) array, same shape as signal. Edge channels
        (within n_stack // 2 of either boundary) are stacked over a
        boundary-extended window (scipy's mode='nearest'), not padded with
        zeros or dropped, so the same distance_km axis still applies with no
        boundary special-casing needed downstream.

    Raises:
        ValueError: If n_stack < 1 or signal is not 2D.
    """
    if signal.ndim != 2:
        raise ValueError(f"stack_local_channels() expects a 2D (n_samples, n_channels) array, got ndim={signal.ndim}")
    if n_stack < 1:
        raise ValueError(f"n_stack must be >= 1, got {n_stack}")
    if mode not in ("mean", "sum"):
        raise ValueError(f"mode must be 'mean' or 'sum', got {mode!r}")

    stacked = uniform_filter1d(signal, size=n_stack, axis=1, mode='nearest')
    if mode == "sum":
        stacked = stacked * n_stack
    return stacked


def preprocess_channels(
    strain_rate: np.ndarray,
    fs: float,
    bandpass: Tuple[float, float],
    distance_km: Union[float, np.ndarray],
    baseline_curve: np.poly1d,
    channel_spacing: float,
    bad_channel_action: str = "interpolate",
    order: int = 4,
    fk_c_low: float = 1400.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full raw-to-model-ready preprocessing pipeline for one or many channels.

    Composes detect_bad_channels() -> (interpolate_bad_channels() or leave
    as-is) -> remove_common_mode() -> detrend_bandpass() -> fk_filter()
    (highpass) -> normalize_by_distance(). This is the one pipeline call
    every notebook location that needs fully processed signal uses (Section
    2's EDA, Section 3's single-channel demo AND batch feature extraction,
    Section 7's flagged-window view), so none of these steps can silently
    diverge between call sites -- including what actually gets fed to the
    anomaly-detection models, not just what gets plotted.

    remove_common_mode()/fk_filter() only apply when strain_rate is 2D
    (multi-channel) -- both are inherently multi-channel operations (nothing
    to call "common" or no spatial axis to FFT for a single channel), so
    they're a no-op for a 1D single-channel call.

    Args:
        strain_rate: (n_samples,) or (n_samples, n_channels) raw strain rate.
        fs: Sampling rate, Hz.
        bandpass: (low, high) cutoff frequencies, Hz.
        distance_km: scalar or per-channel distance from the deep-water
            start, passed to normalize_by_distance().
        baseline_curve: fitted distance-baseline RMS curve (notebook Section 2f).
        channel_spacing: EFFECTIVE distance between adjacent columns of
            strain_rate, metres -- passed to fk_filter(). If strain_rate's
            channels were spatially subsampled (e.g. Section 3a-ii's
            CHANNEL_STRIDE), this must be CHANNEL_STRIDE * CHANNEL_SPACING_M
            (the real distance between adjacent SELECTED channels), not the
            underlying array's native spacing -- fk_filter()'s
            apparent-velocity computation depends on this being correct.
            Unused (but still required) for 1D input.
        bad_channel_action: "interpolate" (average nearest good neighbours --
            only valid for a contiguous channel range) or "skip" (leave bad
            channels in the filtered output; the caller drops them using the
            returned mask -- for strided/non-contiguous channel selections
            where interpolation isn't physically meaningful).
        order: Butterworth filter order.
        fk_c_low: fk_filter()'s lower apparent-velocity bound, m/s -- a
            highpass, not a band (see fk_filter()'s docstring for why).
            Unused for 1D input.

    Returns:
        (processed, is_bad): processed is the filtered + distance-normalized
        signal, same shape as strain_rate; is_bad is the boolean mask from
        detect_bad_channels().

    Raises:
        ValueError: If bad_channel_action is not "interpolate" or "skip".
    """
    if bad_channel_action not in ("interpolate", "skip"):
        raise ValueError(
            f"bad_channel_action must be 'interpolate' or 'skip', got {bad_channel_action!r}"
        )

    is_bad = detect_bad_channels(strain_rate)
    if bad_channel_action == "interpolate" and strain_rate.ndim == 2 and is_bad.any():
        strain_rate = interpolate_bad_channels(strain_rate, is_bad)

    if strain_rate.ndim == 2:
        strain_rate = remove_common_mode(strain_rate)

    filtered = detrend_bandpass(strain_rate, fs, bandpass, order)

    if filtered.ndim == 2:
        filtered = fk_filter(filtered, fs, channel_spacing, c_low=fk_c_low)

    processed = normalize_by_distance(filtered, distance_km, baseline_curve)
    return processed, is_bad


# ============================================================
# Self-check (shared by notebook pre-flight gate + unit tests)
# ============================================================

def preprocess_self_check() -> None:
    """Run the preprocessing pipeline on a small synthetic multi-channel array.

    Verifies detect_bad_channels() flags a NaN channel, interpolate_bad_channels()
    removes it, and the full preprocess_channels() pipeline preserves shape and
    finiteness through detrend + bandpass + distance normalization. Used as a
    fast pre-flight gate before any real data loading (notebook Section 0), same
    role as dsp_features.self_check() but for the preprocessing half of the
    pipeline.

    Raises:
        AssertionError: If any sanity check fails.
    """
    fs, n_samples = 250.0, 5_000
    t = np.arange(n_samples, dtype=np.float64) / fs
    sig = np.sin(2 * np.pi * 15 * t)

    multi = np.column_stack([sig, sig, np.full(n_samples, np.nan)])
    is_bad = detect_bad_channels(multi)
    assert is_bad.tolist() == [False, False, True], "detect_bad_channels() didn't flag the NaN channel"
    clean = interpolate_bad_channels(multi, is_bad)
    assert np.all(np.isfinite(clean)), "interpolate_bad_channels() left non-finite values"

    processed, _ = preprocess_channels(
        multi, fs, (2, 100), distance_km=np.array([0.0, 1.0, 2.0]),
        baseline_curve=np.poly1d([1.0]), channel_spacing=16.8,
        bad_channel_action="interpolate",
    )
    assert processed.shape == multi.shape, "preprocess_channels() changed array shape"
    assert np.all(np.isfinite(processed)), "preprocess_channels() produced non-finite output"

    # Single-channel call (no interpolation branch, distance_km scalar) must
    # also work -- e.g. a single representative channel sliced from a batch
    # result. channel_spacing is required but unused for 1D input (no
    # common-mode/fk step runs).
    single_processed, single_is_bad = preprocess_channels(
        sig, fs, (2, 100), distance_km=0.0,
        baseline_curve=np.poly1d([1.0]), channel_spacing=16.8,
        bad_channel_action="skip",
    )
    assert single_processed.shape == sig.shape, "preprocess_channels() single-channel shape mismatch"
    assert single_is_bad.shape == (1,), "preprocess_channels() single-channel is_bad shape mismatch"

    # remove_common_mode(): a shared component present on every channel must
    # be removed, leaving each channel's own (here: zero) residual.
    rng = np.random.default_rng(0)
    common_component = np.sin(2 * np.pi * 3 * t)
    shared = np.column_stack([common_component] * 4)
    de_commoned = remove_common_mode(shared, method="median")
    assert de_commoned.shape == shared.shape, "remove_common_mode() changed array shape"
    assert np.allclose(de_commoned, 0.0, atol=1e-9), "remove_common_mode() didn't remove an identical shared component"

    # fk_filter(): a plane wave at a known apparent velocity must survive a
    # pass band that includes it, and be strongly attenuated by one that
    # excludes it.
    # n_t_fk chosen long enough (20s) that the default 2.0s edge taper stays a
    # small fraction of the record, matching realistic usage -- a signal only
    # slightly longer than the taper duration would be over-attenuated by the
    # taper itself, not by the f-k mask under test.
    n_t_fk, n_x_fk, fs_fk, dx_fk = 5_000, 32, 250.0, 16.8
    t_fk = np.arange(n_t_fk) / fs_fk
    x_fk = np.arange(n_x_fk) * dx_fk
    c_true = 1490.0
    plane_wave = np.sin(2 * np.pi * 10 * (t_fk[:, None] - x_fk[None, :] / c_true))
    passed = fk_filter(plane_wave, fs_fk, dx_fk, c_low=1000.0, c_high=2000.0)
    rejected = fk_filter(plane_wave, fs_fk, dx_fk, c_low=50.0, c_high=200.0)
    assert passed.shape == plane_wave.shape, "fk_filter() changed array shape"
    assert np.all(np.isfinite(passed)), "fk_filter() produced non-finite output"
    passed_energy = np.sum(passed ** 2)
    rejected_energy = np.sum(rejected ** 2)
    original_energy = np.sum(plane_wave ** 2)
    assert passed_energy > 0.5 * original_energy, "fk_filter() over-attenuated a band that should pass the true apparent velocity"
    assert rejected_energy < 0.1 * original_energy, "fk_filter() under-attenuated a band that should reject the true apparent velocity"

    # Default (c_high=None) must behave as a highpass: pass when c_low is
    # below the true apparent velocity, reject when c_low is above it.
    passed_highpass = fk_filter(plane_wave, fs_fk, dx_fk, c_low=1000.0)
    rejected_highpass = fk_filter(plane_wave, fs_fk, dx_fk, c_low=2000.0)
    assert np.sum(passed_highpass ** 2) > 0.5 * original_energy, "fk_filter() highpass (c_high=None) over-attenuated below the true apparent velocity"
    assert np.sum(rejected_highpass ** 2) < 0.1 * original_energy, "fk_filter() highpass (c_high=None) under-attenuated above the true apparent velocity"

    # stack_local_channels(): shape/finiteness for both odd and even n_stack,
    # and stacking must reduce independent per-channel noise (lower std).
    noisy = rng.standard_normal((2_000, 20))
    for n_stack in (3, 4):
        stacked = stack_local_channels(noisy, n_stack=n_stack, mode="mean")
        assert stacked.shape == noisy.shape, f"stack_local_channels(n_stack={n_stack}) changed array shape"
        assert np.all(np.isfinite(stacked)), f"stack_local_channels(n_stack={n_stack}) produced non-finite output"
    assert stack_local_channels(noisy, n_stack=5, mode="mean").std() < noisy.std(), \
        "stack_local_channels() didn't reduce independent per-channel noise"


# ============================================================
# Quick test
# ============================================================
if __name__ == '__main__':
    preprocess_self_check()
    print("preprocess_self_check() OK")
