"""Core CovGate algorithm: eigenvalue-weighted covariance gating.

Implements the Minimal Intervention Principle for black-box stochastic policies:
suppress variability on action axes where K samples agree (task-relevant),
pass through variability on axes where samples disagree (task-irrelevant).
No retraining, no architecture change — operates purely on policy output samples.
"""

from __future__ import annotations

import numpy as np

from covgate.cluster import commit_cluster


def select_medoid(samples: np.ndarray) -> np.ndarray:
    """Return the single sample with the lowest total distance to all others.

    The medoid is the most "agreed-upon" individual sample.  Unlike the
    cross-sample mean (condition B), it is always a valid action sequence.

    Args:
        samples: (K, T, A) array of K action-chunk samples.

    Returns:
        medoid: (T, A) action chunk.
    """
    K = samples.shape[0]
    flat = samples.reshape(K, -1)
    dists = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)
    return samples[int(np.argmin(dists.sum(axis=1)))]


def gate_action(
    samples: np.ndarray,
    gamma: float = 1.0,
    abs_floor: float = 0.0,
    anchor: str = "first",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply eigenvalue-weighted covariance gating to K action-chunk samples.

    For each timestep τ within the chunk horizon:
      1. Compute the K×K cross-sample covariance Σ_τ.
      2. Eigendecompose Σ_τ = V Λ Vᵀ.
      3. Gate each eigenaxis i with g_i = (λ_i / λ_max)^γ.
         - Low eigenvalue (samples agree) → g_i → 0 → clamp toward mean.
         - High eigenvalue (samples disagree) → g_i → 1 → pass raw sample.
      4. Reconstruct: a_τ = ā_τ + V diag(g) Vᵀ (a_raw_τ − ā_τ).

    This is the core operation from the paper (Eq. 1–2).

    Args:
        samples: (K, T, A) array of K independent action-chunk samples drawn
            from the same policy and observation.
        gamma: gate sharpness exponent. gamma=1 (default) is linear.
            gamma>1 sharpens (harder clamp); gamma<1 softens (more passthrough).
        abs_floor: absolute eigenvalue threshold. Axes with λ_i below this
            value are force-clamped to mean regardless of their relative rank.
            Set to 0 (default) to disable.
        anchor: which deviation source to use.
            "first" (default): samples[0] deviation — original behaviour.
            "medoid": most central sample's deviation.
            "fitted": draw deviation from estimated N(0, Λ) in eigenbasis;
                K-convergent because λ_i converges as K → ∞.
        rng: random number generator used when anchor="fitted".
            None (default) uses np.random.default_rng() (system entropy).

    Returns:
        action: (T, A) gated action chunk.
    """
    K, T, A = samples.shape
    mean = samples.mean(axis=0)
    out = np.empty_like(mean)

    if anchor == "fitted":
        _rng = rng if rng is not None else np.random.default_rng()
    else:
        raw = samples[0] if anchor == "first" else select_medoid(samples)

    for t in range(T):
        X = samples[:, t, :]
        cov = np.cov(X, rowvar=False)
        if A == 1:
            cov = cov.reshape(1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        max_eig = eigvals.max()
        gate = (eigvals / max_eig) ** gamma if max_eig > 1e-12 else np.zeros_like(eigvals)
        if abs_floor > 0.0:
            gate = np.where(eigvals < abs_floor, 0.0, gate)
        if anchor == "fitted":
            z = _rng.standard_normal(A)
            out[t] = mean[t] + eigvecs @ (gate * np.sqrt(eigvals) * z)
        else:
            dev = raw[t] - mean[t]
            out[t] = mean[t] + eigvecs @ (gate * (eigvecs.T @ dev))

    return out


def gate_action_two_stage(
    samples: np.ndarray,
    gamma: float = 1.0,
    abs_floor: float = 0.0,
    stage1_window: int = 1,
    stage2: str = "gate",
    commit_threshold: float = 0.0,
    anchor: str = "first",
    rng: np.random.Generator | None = None,
    adaptive_window: bool = False,
    majority_select: bool = False,
    anchor_mode: str = "window",
) -> np.ndarray:
    """Two-stage CovGate: plan-commitment (Stage 1) then covariance gating (Stage 2).

    Stage 1 clusters the K samples on an early action window (encoding plan
    identity) and selects the most internally-consistent cluster.  Stage 2
    applies covariance gating within that committed subset.

    This is condition E2 from the paper (stage1_window=1, stage2="gate"),
    which Pareto-dominates all other conditions on both PushT and BlockPush.

    Args:
        samples: (K, T, A) array of K action-chunk samples.
        gamma: gate sharpness exponent passed to gate_action.
        abs_floor: absolute eigenvalue floor passed to gate_action.
        stage1_window: number of leading timesteps used for plan-commitment
            clustering. 1 = first timestep only (E2). 0 or >=T = full chunk (E).
        stage2: what to do within the committed cluster.
            "gate" (default): apply gate_action (condition E / E2).
            "medoid": select the medoid of the committed cluster (condition E3).
        commit_threshold: R² threshold for variance-ratio guard passed to
            commit_cluster. 0.0 (default) = original E2 behaviour unchanged.
            Set to 0.3 for E2v (skip commitment on unimodal inputs).
        anchor: deviation source for Stage-2; passed to gate_action.
            "first" (default) | "medoid" | "fitted". See gate_action.
        rng: random generator for anchor="fitted"; passed to gate_action.
        adaptive_window: if True, use adaptive-window clustering (M1).
        majority_select: if True, select majority cluster at Stage 1 (M2).

    Returns:
        action: (T, A) gated action chunk.
    """
    committed = commit_cluster(
        samples, stage1_window, commit_threshold,
        adaptive_window=adaptive_window, majority_select=majority_select,
        anchor_mode=anchor_mode,
    )
    if stage2 == "gate":
        return gate_action(committed, gamma=gamma, abs_floor=abs_floor,
                           anchor=anchor, rng=rng)
    elif stage2 == "medoid":
        return select_medoid(committed)
    else:
        raise ValueError(f"unknown stage2: {stage2!r}")


class PersistentCovGate:
    """Stateful CovGate that carries commitment context across episode timesteps (M3).

    At each call the cluster selection is biased toward the cluster whose
    step-0 centroid is closest to the previously committed cluster's step-0
    centroid, implementing plan continuity (M3).  On the first call
    (or after reset()) falls back to majority_select / variance selection.

    Args:
        gamma: eigenvalue gate exponent passed to gate_action.
        abs_floor: absolute eigenvalue floor passed to gate_action.
        stage1_window: clustering window passed to commit_cluster.
        adaptive_window: enable M1 adaptive window selection.
        majority_select: enable M2 majority cluster selection.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        abs_floor: float = 0.0,
        stage1_window: int = 1,
        adaptive_window: bool = False,
        majority_select: bool = False,
    ) -> None:
        self.gamma = gamma
        self.abs_floor = abs_floor
        self.stage1_window = stage1_window
        self.adaptive_window = adaptive_window
        self.majority_select = majority_select
        self._prev_centroid: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_centroid = None

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """Gate samples with M3 plan-continuity cluster selection.

        Args:
            samples: (K, T, A) array of action-chunk samples.

        Returns:
            action: (T, A) gated action chunk.
        """
        committed = commit_cluster(
            samples,
            stage1_window=self.stage1_window,
            adaptive_window=self.adaptive_window,
            majority_select=self.majority_select,
            prev_centroid=self._prev_centroid,
        )
        self._prev_centroid = committed[:, 0, :].mean(axis=0).copy()
        return gate_action(committed, gamma=self.gamma, abs_floor=self.abs_floor)
