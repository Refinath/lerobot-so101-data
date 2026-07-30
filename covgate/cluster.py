"""Stage-1 plan-commitment clustering for the two-stage CovGate architecture."""

from __future__ import annotations

import numpy as np


def cluster_two_means(flat: np.ndarray) -> np.ndarray:
    """Deterministic 2-means on an (K, D) feature matrix.

    Seeds from the two maximally-distant samples rather than random
    initialization, making the result reproducible without a random seed.

    Args:
        flat: (K, D) array of K samples in D-dimensional feature space.

    Returns:
        labels: (K,) integer array in {0, 1}.
    """
    K = flat.shape[0]
    if K < 2:
        return np.zeros(K, dtype=int)
    dists_sq = np.sum((flat[:, None] - flat[None, :]) ** 2, axis=-1)
    i, j = np.unravel_index(np.argmax(dists_sq), dists_sq.shape)
    centroids = flat[[i, j]].copy()
    labels = np.zeros(K, dtype=int)
    for _ in range(30):
        d0 = np.sum((flat - centroids[0]) ** 2, axis=-1)
        d1 = np.sum((flat - centroids[1]) ** 2, axis=-1)
        new_labels = (d1 < d0).astype(int)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in (0, 1):
            if (labels == c).any():
                centroids[c] = flat[labels == c].mean(axis=0)
    return labels


def commit_cluster(
    samples: np.ndarray,
    stage1_window: int = 1,
    commit_threshold: float = 0.0,
    adaptive_window: bool = False,
    majority_select: bool = False,
    prev_centroid: np.ndarray | None = None,
    anchor_mode: str = "window",
) -> np.ndarray:
    """Stage 1: 2-means plan-commitment with optional variance-ratio guard.

    When commit_threshold > 0, commitment is skipped when the 2-means split
    explains less than commit_threshold of the total sample variance (R² test).
    commit_threshold=0.0 (default) reproduces the original behaviour exactly.

    Args:
        samples: (K, T, A) array of K action-chunk samples.
        stage1_window: number of leading timesteps used as clustering features
            when adaptive_window=False. 1 = first timestep only (E2 default).
        commit_threshold: R² threshold for variance-ratio guard.
        adaptive_window: if True, find the timestep t* with maximum inter-sample
            variance and cluster on steps 0..t* instead of stage1_window steps.
            This adapts the commitment window to where plans diverge in the chunk.
        majority_select: if True, commit to the cluster with more samples (MAP
            plan) rather than the lower-variance cluster. Ties broken by variance.
        prev_centroid: (A,) step-0 action centroid of the previous committed
            cluster. When provided, overrides adaptive_window/majority_select for
            cluster selection: the cluster whose step-0 centroid is closest to
            prev_centroid is chosen (plan continuity). When None, uses
            majority_select or variance as normal (M3 first-step fallback).

    Returns:
        committed: (K_c, T, A) subset of samples belonging to the committed
            cluster. Falls back to the full set when K < 2 or degenerate.
    """
    K, T, A = samples.shape
    if K < 2:
        return samples

    # Determine clustering anchor.
    if anchor_mode == "mean":
        flat_cluster = samples.mean(axis=1)                             # (K, A)
    elif anchor_mode == "spread":
        per_step_var = samples.var(axis=0).mean(axis=-1)               # (T,)
        w_t = per_step_var / (per_step_var.sum() + 1e-8)
        flat_cluster = (samples * w_t[None, :, None]).sum(axis=1)      # (K, A)
    elif anchor_mode == "traj":
        flat_cluster = samples.reshape(K, -1)                          # (K, T*A)
    elif anchor_mode == "pc1":
        _flat = samples.reshape(K, -1).astype(float)
        _flat_c = _flat - _flat.mean(axis=0)
        _G = _flat_c @ _flat_c.T / max(K - 1, 1)                      # (K, K)
        _eigvals, _eigvecs = np.linalg.eigh(_G)
        flat_cluster = (_eigvecs[:, -1:] * _eigvals[-1]).astype(       # (K, 1)
            samples.dtype)
    else:  # "window" — original behaviour, backward compatible
        if adaptive_window:
            per_step_var = samples.var(axis=0).mean(axis=-1)           # (T,)
            t_star = int(per_step_var.argmax())
            w = max(1, t_star + 1)
        else:
            w = stage1_window if (stage1_window and 0 < stage1_window < T) else T
        flat_cluster = samples[:, :w, :].reshape(K, -1)
    flat_full = samples.reshape(K, -1)
    labels = cluster_two_means(flat_cluster)

    # Optional R² guard (unchanged from original).
    if commit_threshold > 0.0:
        v_total = float(np.mean(np.var(flat_full, axis=0)))
        v_within = 0.0
        for c in (0, 1):
            mask = labels == c
            if mask.sum() >= 2:
                v_within += mask.mean() * float(
                    np.mean(np.var(flat_full[mask], axis=0))
                )
        r2 = (v_total - v_within) / v_total if v_total > 1e-12 else 0.0
        if r2 < commit_threshold:
            return samples

    # Select cluster.
    best_cluster = None

    if prev_centroid is not None:
        # M3: proximity to previous commitment in step-0 action space.
        best_dist = np.inf
        for c in (0, 1):
            mask = labels == c
            if mask.sum() < 2:
                continue
            centroid_c = samples[mask, 0, :].mean(axis=0)
            d = float(np.linalg.norm(centroid_c - prev_centroid))
            if d < best_dist:
                best_dist, best_cluster = d, c

    if best_cluster is None:
        if majority_select:
            # M2: commit to the cluster with more samples.
            best_count, best_var_tie = -1, np.inf
            for c in (0, 1):
                mask = labels == c
                count = int(mask.sum())
                if count < 2:
                    continue
                if count > best_count:
                    best_count, best_cluster = count, c
                elif count == best_count:
                    var = float(np.mean(np.var(flat_full[mask], axis=0)))
                    if var < best_var_tie:
                        best_var_tie, best_cluster = var, c
        else:
            # Original: commit to the cluster with lowest intra-cluster variance.
            best_var = np.inf
            for c in (0, 1):
                mask = labels == c
                if mask.sum() < 2:
                    continue
                var = float(np.mean(np.var(flat_full[mask], axis=0)))
                if var < best_var:
                    best_var, best_cluster = var, c

    min_size = max(2, K // 2) if anchor_mode in ("mean", "spread", "traj", "pc1") else 2
    if best_cluster is None or (labels == best_cluster).sum() < min_size:
        return samples
    return samples[labels == best_cluster]
