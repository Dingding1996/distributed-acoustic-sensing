"""
Valencia-IslaLink DAS Dataset - Unsupervised Anomaly Detection
=================================================================
Isolation Forest, wrapped in a sklearn Pipeline (StandardScaler + model) so
the scaler is re-fit inside every fold. No labels anywhere -- there is no
ground truth for this dataset (CLAUDE.md SS1).

One-Class SVM was dropped (not adapted) after its libsvm-backed fit --
subsampled to 5000 of ~846K training rows for tractable runtime -- turned
out to correlate weakly with both Isolation Forest and the hand-crafted
baseline detector (Spearman rho ~0.58 vs their ~0.92 with each other),
while this project isn't ML-architecture-focused enough to justify chasing
that gap (e.g. raising the subsample size, which reintroduces the runtime
problem it was added to avoid). `fuse_scores()`'s two-model ensemble went
with it -- nothing left to fuse once only one ML model remains; Section 6
compares Isolation Forest against the independent, non-ML baseline detector
instead (CLAUDE.md SS9).

Replaces the predecessor bearing-fault project's utils/ml_classification.py
(TraditionalMLPipeline: RF/GBT/XGBoost, supervised) -- a clean replacement,
not an adaptation, since nothing in that module's fitting/evaluation logic
applies without labels.
"""

import numpy as np
from typing import Dict, Iterator, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


class AnomalyDetectionPipeline:
    """Registers Isolation Forest in a Pipeline.

    Mirrors the predecessor project's TraditionalMLPipeline shape (a class
    registering named sklearn Pipeline objects) but adapted for unsupervised
    fitting -- no y anywhere, no accuracy/F1, continuous anomaly_scores
    instead of class predictions. A dict-of-one-pipeline shape (rather than a
    single bare Pipeline) is kept deliberately, not simplified away -- it's
    what lets score()/predict_anomaly() stay generic over "however many
    models are registered" instead of hardcoding "IsolationForest" into
    every caller.
    """

    def __init__(
        self,
        if_n_estimators: int = 100,
        if_contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.pipelines: Dict[str, Pipeline] = {
            "IsolationForest": Pipeline([
                ("scaler", StandardScaler()),
                ("model", IsolationForest(
                    n_estimators=if_n_estimators,
                    contamination=if_contamination,
                    random_state=random_state,
                    n_jobs=-1,
                )),
            ]),
        }
        self.random_state = random_state
        self.fitted_pipelines: Dict[str, Pipeline] = {}
        self.train_scores_: Dict[str, np.ndarray] = {}

    def fit(self, X_train: np.ndarray) -> "AnomalyDetectionPipeline":
        """Fit every registered pipeline on X_train. No y -- unsupervised.

        Args:
            X_train: Training feature matrix, (n_samples, n_features).

        Returns:
            self, for chaining.
        """
        for name, pipe in self.pipelines.items():
            pipe.fit(X_train)
        self.fitted_pipelines = dict(self.pipelines)
        self.train_scores_ = self.score(X_train)
        return self

    def score(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Continuous anomaly score per fitted model -- higher = more anomalous.

        IsolationForest.decision_function returns higher = more normal
        natively; sign is flipped here so callers never have to remember
        this convention.

        Args:
            X: Feature matrix to score, (n_samples, n_features).

        Returns:
            Dict of {model_name: anomaly_scores}.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.fitted_pipelines:
            raise RuntimeError("fit() must be called before score()")
        return {name: -pipe.decision_function(X) for name, pipe in self.fitted_pipelines.items()}

    def predict_anomaly(self, X: np.ndarray, contamination: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Boolean is_anomaly_pred per model -- model output, not ground truth.

        Not called from the training notebook any more (CLAUDE.md SS9): a
        fixed contamination rate is itself an unverifiable guess with no
        ground truth to check it against, so the notebook now reports
        continuous scores and rank-based top-N selections instead. Kept here
        (not deleted, same as `stack_local_channels()` in
        `data_preprocess.py`) since a future inference API consumer may
        still want a simple boolean cutoff rather than a raw score.

        Thresholds each model's scores on X at the percentile of that SAME
        model's *training* scores implied by `contamination` -- the cutoff is
        always derived from training rows, never from X itself, so scoring a
        different X (e.g. a held-out event) can't move its own threshold.

        Args:
            X: Feature matrix to flag, (n_samples, n_features).
            contamination: Dict of {model_name: assumed anomalous fraction},
                e.g. {"IsolationForest": 0.05}.

        Returns:
            Dict of {model_name: is_anomaly_pred} (boolean arrays).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self.fitted_pipelines:
            raise RuntimeError("fit() must be called before predict_anomaly()")
        scores = self.score(X)
        is_anomaly_pred = {}
        for name, s in scores.items():
            cutoff = np.percentile(self.train_scores_[name], 100 * (1 - contamination[name]))
            is_anomaly_pred[name] = s > cutoff
        return is_anomaly_pred


def chronological_group_splits(event_ids_sorted: List[str]) -> Iterator[Tuple[List[str], str]]:
    """Walk-forward chronological splits: train on all earlier events, test on the next one.

    No sklearn splitter does grouped + chronological together (`TimeSeriesSplit`
    ignores groups, `GroupKFold` ignores order) -- this is the minimal custom
    splitter that does (CLAUDE.md SS8.4: last resort, but nothing existing fits).

    Yields one (train_event_ids, test_event_id) pair per event after the
    first -- e.g. 4 sorted events yield 3 splits: ([e1],e2), ([e1,e2],e3),
    ([e1,e2,e3],e4). Only knows about event ordering, not row-level data --
    callers convert event ids to row masks themselves.

    Args:
        event_ids_sorted: Unique event ids, already in chronological order.

    Yields:
        (train_event_ids, test_event_id) for each walk-forward step.
    """
    for i in range(1, len(event_ids_sorted)):
        yield event_ids_sorted[:i], event_ids_sorted[i]


def fit_distance_curves(
    X_train: np.ndarray, distances_train: np.ndarray, feature_names: List[str], degree: int = 3,
) -> Dict[int, Tuple[np.poly1d, np.poly1d, bool]]:
    """Fit per-feature smooth mean AND scale curves against distance.

    The single raw-signal-level distance-baseline curve (Section 2g) only
    partially decorrelates engineered features from distance -- verified
    directly: `crest`/`kurt`/`peakiness` are scale-invariant ratios that a
    signal-amplitude correction can never touch, and `band_*`/`wpd_*` energies
    still carry residual correlation (|r| up to ~0.5) because different
    frequency bands attenuate differently with distance, which one broadband
    RMS curve can't capture per-band.

    Mean-centering alone (subtracting a per-feature mean curve) was tried
    first and verified NOT to fix the actual problem: IsolationForest/
    OneClassSVM flag rates still grew ~50x from near to far distance, because
    residual *variance* keeps growing with distance even after the mean is
    removed (CLAUDE.md SS9's predicted false-positive-rate skew, worse in
    practice than expected). So this fits a scale curve too -- but as a
    smooth fit across many channels' residual variance vs distance, not each
    channel's own noisy local std (the thing CLAUDE.md SS9 originally warned
    against, for being unstable/erratic per-channel, not for the underlying
    idea of scale-correcting by distance).

    Energy-domain features (`band_*`/`wpd_*` prefixes) are fit in sqrt space
    (variance-stabilizing, same reasoning as CLAUDE.md SS9's raw-signal curve);
    everything else (already linear amplitude or a dimensionless ratio) is fit
    directly.

    Args:
        X_train: Training feature matrix, (n_samples, n_features).
        distances_train: Distance (km) per training row, (n_samples,).
        feature_names: Column names for X_train, used only to decide sqrt-space.
        degree: Polynomial degree -- kept low so the curves track the broad
            distance trend without chasing per-window noise.

    Returns:
        Dict of {feature_index: (mean_curve, variance_curve, is_sqrt_space)}.
    """
    curves = {}
    for fi, name in enumerate(feature_names):
        is_sqrt = name.startswith("band_") or name.startswith("wpd_")
        y = np.sqrt(np.clip(X_train[:, fi], 0, None)) if is_sqrt else X_train[:, fi]
        mean_curve = np.poly1d(np.polyfit(distances_train, y, deg=degree))
        residual = y - mean_curve(distances_train)
        variance_curve = np.poly1d(np.polyfit(distances_train, residual ** 2, deg=degree))
        curves[fi] = (mean_curve, variance_curve, is_sqrt)
    return curves


def apply_distance_curves(
    X: np.ndarray, distances: np.ndarray, curves: Dict[int, Tuple[np.poly1d, np.poly1d, bool]],
) -> np.ndarray:
    """Mean-center AND scale-divide each feature by its fitted distance curves.

    Args:
        X: Feature matrix to decorrelate, (n_samples, n_features).
        distances: Distance (km) per row, (n_samples,).
        curves: Output of `fit_distance_curves` -- must be fit on training
            rows only; applying to held-out rows here doesn't refit anything.

    Returns:
        Z-scored residuals per feature (sqrt-space features are returned in
        their residual sqrt-space, not squared back).
    """
    X_resid = X.copy().astype(np.float64)
    for fi, (mean_curve, variance_curve, is_sqrt) in curves.items():
        col = np.sqrt(np.clip(X[:, fi], 0, None)) if is_sqrt else X[:, fi]
        std_fitted = np.sqrt(np.clip(variance_curve(distances), 1e-12, None))
        X_resid[:, fi] = (col - mean_curve(distances)) / std_fitted
    return X_resid
