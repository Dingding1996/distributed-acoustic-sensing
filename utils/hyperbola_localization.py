"""Hyperbola-fit (TDOA) source localization -- extracted from
old/DAS_Signal_Processing.ipynb, not integrated into the current pipeline.

Status: future-phase prep only. Source localization/beamforming is
explicitly out of scope for this revision (CLAUDE.md SS1) -- this module
exists so a later phase doesn't have to re-derive the algorithm from the
archived notebook. Not imported by DAS_Training.ipynb or any other utils/
module, not adapted or validated against the Valencia-IslaLink dataset.

Method: given a detected arrival at one ("peak") channel, cross-correlate a
short template around that arrival against every other channel to estimate
each channel's relative time delay and a normalized-correlation quality
score, then fit a hyperbolic moveout model

    t(x) = t0 + sqrt((x - x0)^2 + z0^2) / c

(a point source at along-cable position x0, perpendicular offset z0,
arriving at reference time t0, propagating at speed c) to the high-quality
picks via least-squares.
"""

from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import correlate


def hyperbola_arrival_time(x: np.ndarray, x0: float, z0: float, t0: float, c: float) -> np.ndarray:
    """Hyperbolic moveout model: predicted arrival time at along-cable position x.

    Args:
        x: Along-cable channel position(s), metres.
        x0: Source position along the cable, metres.
        z0: Source perpendicular offset from the cable, metres.
        t0: Arrival time at x0 (the closest point), seconds.
        c: Propagation speed, m/s.

    Returns:
        Predicted arrival time(s) at x, seconds.
    """
    return t0 + np.sqrt((x - x0) ** 2 + z0 ** 2) / c


def pick_arrival_times(
    signal: np.ndarray,
    distance: np.ndarray,
    fs: float,
    peak_channel: int,
    peak_time_idx: int,
    c_search: float,
    template_half_width_sec: float = 0.06,
    search_margin_sec: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-correlation arrival-time picks (TDOA) relative to one peak channel.

    Cuts a short template around the peak channel's own arrival, then
    cross-correlates it against a search window on every other channel
    (sized from a rough expected time-of-flight at c_search, so the search
    stays local instead of scanning the whole record). The lag of the
    correlation peak gives that channel's relative delay; the normalized
    correlation value at that lag is a quality score (1.0 = perfect match).

    Args:
        signal: (n_samples, n_channels) array, time along axis 0 -- should
            already be filtered to the band the arrival is expected in.
        distance: (n_channels,) along-cable position of each channel, metres.
        fs: Sampling rate, Hz.
        peak_channel: Index (into signal's columns) of the reference channel
            the template is cut from.
        peak_time_idx: Sample index of the reference arrival on peak_channel.
        c_search: Rough expected propagation speed, m/s -- used only to size
            each channel's search window, not the fit itself.
        template_half_width_sec: Half-width of the cross-correlation
            template, seconds.
        search_margin_sec: Extra time padded onto each channel's expected
            arrival window, seconds.

    Returns:
        (delay_time, quality): both (n_channels,) arrays. delay_time is the
        ABSOLUTE arrival time (seconds, same time base as peak_time_idx/fs)
        picked on each channel, NaN where the search window was too short to
        hold a full template. quality is the normalized cross-correlation
        value at the picked lag (NaN at the same entries as delay_time).
    """
    n_samples, n_channels = signal.shape
    half_tmpl = int(template_half_width_sec * fs)
    i_s = max(0, peak_time_idx - half_tmpl)
    i_e = min(n_samples, peak_time_idx + half_tmpl)
    template = signal[i_s:i_e, peak_channel]
    template_rms = np.sqrt(np.mean(template ** 2)) + 1e-30
    x_peak = distance[peak_channel]

    delay_time = np.full(n_channels, np.nan)
    quality = np.full(n_channels, np.nan)
    for i in range(n_channels):
        dx = abs(distance[i] - x_peak)
        expected_travel_sec = dx / c_search + search_margin_sec
        i0 = max(0, peak_time_idx - int(expected_travel_sec * fs))
        i1 = min(n_samples, peak_time_idx + int(expected_travel_sec * fs))
        segment = signal[i0:i1, i]
        if len(segment) < len(template):
            continue
        cc = correlate(segment, template, mode="valid")
        lag = int(np.argmax(cc))
        window = segment[lag: lag + len(template)]
        window_rms = np.sqrt(np.mean(window ** 2)) + 1e-30
        quality[i] = cc[lag] / (len(template) * window_rms * template_rms)
        delay_time[i] = (i0 + lag + half_tmpl) / fs

    return delay_time, quality


def fit_hyperbola(
    distance: np.ndarray,
    arrival_time: np.ndarray,
    quality: np.ndarray,
    c: float,
    quality_threshold: float = 0.8,
    z0_bounds: Tuple[float, float] = (10.0, 5000.0),
    x0_margin_m: float = 2000.0,
) -> dict:
    """Fit the hyperbolic moveout model to cross-correlation arrival picks.

    Keeps only picks with quality >= quality_threshold, auto-estimates a z0
    starting guess from pick geometry (z0 ~= dx^2 / (2 c dt), the
    small-angle approximation), then jointly fits (x0, z0, t0) via
    least-squares (scipy.optimize.curve_fit).

    Args:
        distance: (n_channels,) along-cable position of each channel, metres.
        arrival_time: (n_channels,) absolute arrival time picks, seconds
            (e.g. pick_arrival_times()'s delay_time output) -- NaN entries
            are excluded automatically.
        quality: (n_channels,) pick quality scores, same shape/NaN pattern.
        c: Propagation speed used in the model, m/s.
        quality_threshold: Minimum quality score for a pick to be used.
        z0_bounds: (min, max) bounds on the fitted perpendicular offset,
            metres.
        x0_margin_m: How far outside the observed channel range x0 is
            allowed to be fit, metres.

    Returns:
        Dict with keys x0, z0, t0 (fitted parameters), x0_err/z0_err/t0_err
        (1-sigma parameter uncertainties from the fit covariance),
        rms_residual_ms (fit residual, milliseconds), and n_picks (number of
        picks used).

    Raises:
        ValueError: If fewer than 3 picks pass quality_threshold (curve_fit
            needs at least as many points as free parameters).
    """
    good = ~np.isnan(arrival_time) & (quality >= quality_threshold)
    if good.sum() < 3:
        raise ValueError(
            f"Only {int(good.sum())} pick(s) pass quality_threshold={quality_threshold} "
            "-- need at least 3 to fit (x0, z0, t0)."
        )

    x_good = distance[good]
    t_good = arrival_time[good]
    peak_idx = np.argmin(t_good)  # earliest-arriving pick as the reference point
    t_peak, x_peak = t_good[peak_idx], x_good[peak_idx]

    dx = np.abs(x_good - x_peak)
    dt = t_good - t_peak  # >= 0 by construction (t_peak is the earliest pick)
    valid = dt > 0.002  # exclude near-zero delays (unstable for the z0 estimate below)
    if valid.sum() >= 2:
        z0_init = float(np.median(dx[valid] ** 2 / (2 * c * dt[valid])))
        z0_init = float(np.clip(z0_init, *z0_bounds))
    else:
        z0_init = float(np.mean(z0_bounds))

    def _model(x, x0, z0, t0):
        return hyperbola_arrival_time(x, x0, z0, t0, c)

    popt, pcov = curve_fit(
        _model, x_good, t_good,
        p0=[x_peak, z0_init, t_peak - z0_init / c],
        bounds=(
            [distance.min() - x0_margin_m, z0_bounds[0], t_good.min() - 1.0],
            [distance.max() + x0_margin_m, z0_bounds[1], t_peak],
        ),
        maxfev=10000,
    )
    x0_fit, z0_fit, t0_fit = popt
    perr = np.sqrt(np.diag(pcov))
    rms_residual_ms = float(np.std(t_good - hyperbola_arrival_time(x_good, *popt, c)) * 1000)

    return {
        "x0": float(x0_fit), "z0": float(z0_fit), "t0": float(t0_fit),
        "x0_err": float(perr[0]), "z0_err": float(perr[1]), "t0_err": float(perr[2]),
        "rms_residual_ms": rms_residual_ms,
        "n_picks": int(good.sum()),
    }
