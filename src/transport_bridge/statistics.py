"""Discovery-only coordinates and paired uncertainty."""

import numpy as np

from .common import SEED


def cells(rows):
    return np.array([2 * r["wording"] + r["order"] for r in rows])


def fit_reference(raw, rows):
    # raw shape [baseline, example, head, endpoint]; standardize logit effects.
    x = np.asarray(raw[..., 0], dtype=np.float64)
    labels = cells(rows)
    means = np.stack([x[:, labels == c].mean(1) for c in range(4)], 1)
    residual = x - means[:, labels]
    scale = np.maximum(np.sqrt(np.mean(residual**2, axis=1)), 0.1)
    centers = means / scale[:, None]
    return {"scale": scale.tolist(), "centers": centers.tolist()}


def axis_coordinates(raw, rows, fit, factor):
    x = np.asarray(raw[..., 0], dtype=np.float64) / np.array(fit["scale"])[:, None]
    center = np.array(fit["centers"])
    out = np.empty(x.shape[:2])
    for i, r in enumerate(rows):
        if factor == "wording":
            a, b = r["order"], 2 + r["order"]
        else:
            a, b = 2 * r["wording"], 2 * r["wording"] + 1
        axis = center[:, b] - center[:, a]
        out[:, i] = np.sum((x[:, i] - (center[:, a] + center[:, b]) / 2) * axis, axis=1) / np.sum(
            axis**2, axis=1
        )
    return out


def fit_predictor(states, raw, rows, fit, factor):
    x = np.asarray(states, np.float64)
    y = axis_coordinates(raw, rows, fit, factor).mean(0)
    mx, my = x.mean(0), y.mean()
    xc = x - mx
    alpha = 0.01 * np.mean(np.sum(xc**2, axis=1))
    w = xc.T @ np.linalg.solve(xc @ xc.T + alpha * np.eye(len(x)), y - my)
    return {
        "mean_x": mx.tolist(),
        "mean_y": float(my),
        "weight": w.tolist(),
        "direction": (w / np.linalg.norm(w)).tolist(),
        "alpha": float(alpha),
        "cell_means": [float(y[cells(rows) == c].mean()) for c in range(4)],
    }


def prediction_scores(states, raw, rows, fit, predictor, factor):
    y = axis_coordinates(raw, rows, fit, factor).mean(0)
    p = (np.asarray(states) - predictor["mean_x"]) @ predictor["weight"] + predictor["mean_y"]
    q = np.asarray(predictor["cell_means"])[cells(rows)]
    return {
        name: {
            "r2": float(1 - np.mean((values - y) ** 2) / np.var(y)),
            "mse": float(np.mean((values - y) ** 2)),
        }
        for name, values in [("internal", p), ("prompt_labels", q)]
    }


def geometry(edited, clean, donor_clean, rows, fit, factor):
    scale, centers = np.array(fit["scale"]), np.array(fit["centers"])
    ds = (edited[..., 0] - clean[..., 0]) / scale[:, None]
    out = {
        k: np.empty(ds.shape[:2])
        for k in ["score", "orthogonal", "donor_distance", "recipient_distance"]
    }
    for i, r in enumerate(rows):
        a = 2 * r["wording"] + r["order"]
        b = (
            2 * (1 - r["wording"]) + r["order"]
            if factor == "wording"
            else 2 * r["wording"] + 1 - r["order"]
        )
        v = centers[:, b] - centers[:, a]
        norm2 = np.sum(v * v, axis=1)
        if np.any(norm2 < 1e-8):
            raise ValueError("Degenerate discovery axis")
        score = np.sum(ds[:, i] * v, axis=1) / norm2
        out["score"][:, i] = score
        out["orthogonal"][:, i] = np.linalg.norm(ds[:, i] - score[:, None] * v, axis=1) / np.sqrt(
            norm2
        )
        out["recipient_distance"][:, i] = np.linalg.norm(ds[:, i], axis=1) / np.sqrt(norm2)
        out["donor_distance"][:, i] = np.linalg.norm(
            (edited[:, i, :, 0] - donor_clean[:, i, :, 0]) / scale, axis=1
        ) / np.sqrt(norm2)
    return out


def block_interval(values, rows, mask=None, draws=10000, multiplicity=8):
    values = np.asarray(values, dtype=float)
    if mask is None:
        mask = np.ones(len(rows), bool)
    templates = sorted({r["template"] for r, k in zip(rows, mask, strict=True) if k})
    pairs = sorted({r["pair"] for r, k in zip(rows, mask, strict=True) if k})
    grid = np.array(
        [
            [
                values[
                    mask & np.array([r["template"] == t and r["pair"] == p for r in rows])
                ].mean()
                for p in pairs
            ]
            for t in templates
        ]
    )
    if not np.isfinite(grid).all():
        raise ValueError("Incomplete crossed template/name-pair grid")
    rng = np.random.default_rng(SEED)
    ts = rng.integers(len(templates), size=(draws, len(templates)))
    ps = rng.integers(len(pairs), size=(draws, len(pairs)))
    boot = grid[ts[:, :, None], ps[:, None, :]].mean((1, 2))
    tail = 0.05 / (2 * multiplicity)
    return {
        "mean": float(grid.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "simultaneous_ci": np.quantile(boot, [tail, 1 - tail]).tolist(),
        "multiplicity": multiplicity,
        "n": int(mask.sum()),
        "templates": len(templates),
        "pairs": len(pairs),
        "per_template": dict(zip(map(str, templates), grid.mean(1).tolist(), strict=True)),
    }
