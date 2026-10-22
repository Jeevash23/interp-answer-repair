"""Discovery-only scaling and fixed-axis statistics."""

from __future__ import annotations

import numpy as np


def fit_signature(raw, labels, floor=0.1):
    raw, labels = np.asarray(raw, dtype=float), np.asarray(labels)
    if raw.ndim != 2 or not np.isfinite(raw).all() or set(labels) != {0, 1}:
        raise ValueError("Need finite signatures and both families")
    means = np.stack([raw[labels == c].mean(0) for c in (0, 1)])
    residual = raw - means[labels]
    sd = np.sqrt(np.sum(residual**2, axis=0) / (len(labels) - 2))
    scale = np.maximum(sd, floor)
    centroids = means / scale
    axis = centroids[1] - centroids[0]
    if np.dot(axis, axis) < 1e-8:
        raise ValueError("Degenerate family axis; do not report switching")
    return {"scale": scale, "centroids": centroids, "axis": axis, "within_sd": sd}


def classify(raw, fit):
    scaled = np.asarray(raw) / fit["scale"]
    distance = ((scaled[:, None, :] - fit["centroids"][None, :, :]) ** 2).sum(-1)
    # Exact ties receive 0.5 credit later rather than favoring one family.
    predicted = np.argmin(distance, axis=1)
    predicted = np.where(
        np.isclose(distance[:, 0], distance[:, 1], atol=1e-12, rtol=0), -1, predicted
    )
    return predicted


def movement(before, after, recipient_family, fit):
    """0=BABA, 1=ABBA. Positive always means toward the opposite family."""
    if recipient_family not in (0, 1):
        raise ValueError("Unknown family")
    axis = fit["axis"] * (1 if recipient_family == 0 else -1)
    norm2 = np.dot(axis, axis)
    if norm2 < 1e-8:
        raise ValueError("Degenerate family axis")
    delta = (np.asarray(after) - np.asarray(before)) / fit["scale"]
    score = np.dot(delta, axis) / norm2
    orthogonal = np.linalg.norm(delta - score * axis) / np.sqrt(norm2)
    return float(score), float(orthogonal)


def hierarchical_interval(grid, draws, seed):
    """Crossed template × name-pair bootstrap, with roles already averaged in each cell."""
    grid = np.asarray(grid, dtype=float)
    rng = np.random.default_rng(seed)
    t, n = grid.shape
    ti = rng.integers(t, size=(draws, t))
    ni = rng.integers(n, size=(draws, n))
    values = grid[ti[:, :, None], ni[:, None, :]].mean(axis=(1, 2))
    return np.quantile(values, [0.025, 0.975]).tolist()


def conditional_signflip_p(grid):
    """Exact, two-sided diagnostic conditional on the sampled block design."""
    values = np.asarray(grid).ravel()
    if len(values) > 16:
        raise ValueError("Exact diagnostic restricted to small pilot")
    states = np.arange(2 ** len(values))[:, None]
    signs = 2 * ((states >> np.arange(len(values))) & 1) - 1
    null = (signs * values).mean(axis=1)
    return float(np.mean(np.abs(null) >= abs(values.mean()) - 1e-12))
