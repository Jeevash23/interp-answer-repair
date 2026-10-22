"""Descriptive whole-template analysis of the post-hoc Study 5 robustness extension.

The original test templates are reused. These intervals are not a fresh
confirmatory test, and random-control draws are not treated as new examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "experiment-005"
MODELS = ("gpt2-medium", "pythia-410m-deduped")
CONDITIONS = ("canonical", "controlled")
SITES = ("s2", "end")
DOSES = (0.5, 1, 2)
DRAWS = tuple(range(5))
BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 20_260_905
SCORE_FIELDS = ("clean", "baseline", "patched", "clean_logit", "baseline_logit", "patched_logit")
Row = Mapping[str, Any]


def expected_specs() -> dict[str, dict[str, Any]]:
    """The full planned intervention grid, independent of which outcomes occurred."""
    output = {}
    for site in SITES:
        for source in CONDITIONS:
            for target in CONDITIONS:
                meta = {
                    "site": site,
                    "source": source,
                    "target": target,
                    "alpha": 1.0,
                    "kind": "native",
                    "draw": -1,
                }
                output[f"native_{site}_{source}_to_{target}"] = meta
                if source != target:
                    for alpha in DOSES:
                        output[f"matched_{site}_{source}_to_{target}_alpha_{alpha}"] = {
                            **meta,
                            "alpha": float(alpha),
                            "kind": "matched",
                        }
        meta = {
            "site": site,
            "source": "canonical",
            "target": "controlled",
            "alpha": 1.0,
            "draw": -1,
        }
        for control in ("mismatch", "off_target"):
            output[f"control_{site}_{control}"] = {**meta, "kind": control}
        for control in ("random_heads", "random_direction"):
            for draw in DRAWS:
                output[f"control_{site}_{control}_{draw}"] = {
                    **meta,
                    "draw": draw,
                    "kind": control,
                }
    for state in ("baseline", "s2"):
        meta = {
            "site": "s2",
            "source": "canonical",
            "target": "controlled",
            "alpha": 0.0 if state == "baseline" else 1.0,
            "kind": "block",
            "draw": -1,
        }
        output[f"block_selected_{state}"] = dict(meta)
        for draw in DRAWS:
            output[f"block_random_{draw}_{state}"] = {**meta, "draw": draw}
    return output


def validate_rows(rows: Sequence[Row], *, model: str) -> dict[str, dict[str, Row]]:
    """Require all 56 specs, all 160 IDs, and paired, stable metadata before analysis."""
    planned = expected_specs()
    indexed: dict[str, dict[str, Row]] = {spec: {} for spec in planned}
    required = {
        "example_id",
        "template_id",
        "pattern",
        "model",
        "spec",
        "site",
        "source",
        "target",
        "alpha",
        "draw",
        "kind",
        "injected_norm",
        *SCORE_FIELDS,
    }
    metadata: dict[str, tuple[str, str]] = {}
    baselines: dict[tuple[str, str], tuple[float, ...]] = {}
    for row in rows:
        if missing := required - row.keys():
            raise ValueError(f"Missing row fields: {sorted(missing)}")
        if row["model"] != model:
            raise ValueError(f"Model label {row['model']!r} does not match {model!r}")
        spec = row["spec"]
        if spec not in planned:
            raise ValueError(f"Unplanned intervention spec: {spec}")
        example_id = row["example_id"]
        if not isinstance(example_id, str) or not isinstance(row["template_id"], str):
            raise TypeError("Example and template IDs must be strings")
        if example_id in indexed[spec]:
            raise ValueError(f"Duplicate example/spec row: {example_id}, {spec}")
        for key, expected in planned[spec].items():
            if row[key] != expected:
                raise ValueError(f"Inconsistent {key} for {spec}: {row[key]!r} != {expected!r}")
        if row["pattern"] not in {"ABBA", "BABA"}:
            raise ValueError(f"Unknown pattern: {row['pattern']!r}")
        pair_id, suffix = example_id.rsplit("-", 1)
        if not pair_id or suffix != row["pattern"].lower():
            raise ValueError(f"Example ID and pattern disagree: {example_id}")
        current_meta = (row["template_id"], row["pattern"])
        if example_id in metadata and metadata[example_id] != current_meta:
            raise ValueError(f"Template/pattern changes across specs for {example_id}")
        metadata[example_id] = current_meta
        for key in (*SCORE_FIELDS, "injected_norm"):
            if isinstance(row[key], bool) or not isinstance(row[key], (int, float)):
                raise TypeError(f"Non-numeric {key} in {example_id}, {spec}")
            if not math.isfinite(row[key]):
                raise ValueError(f"Non-finite {key} in {example_id}, {spec}")
        if row["injected_norm"] < 0:
            raise ValueError(f"Negative injection norm in {example_id}, {spec}")
        for key in ("clean", "baseline", "patched"):
            if not -1.000001 <= row[key] <= 1.000001:
                raise ValueError(f"Probability difference outside [-1,1]: {key}")
        baseline_key = (example_id, row["target"])
        current_base = tuple(
            float(row[key]) for key in ("clean", "baseline", "clean_logit", "baseline_logit")
        )
        if baseline_key in baselines and not np.allclose(
            baselines[baseline_key], current_base, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(f"Target clean/baseline outcomes change across specs: {baseline_key}")
        baselines[baseline_key] = current_base
        indexed[spec][example_id] = row
    if len(metadata) != 160:
        raise ValueError(f"Expected 160 unique examples, received {len(metadata)}")
    all_ids = set(metadata)
    for spec, group in indexed.items():
        if set(group) != all_ids:
            raise ValueError(f"Missing or mismatched example IDs for {spec}")
    by_template: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for example_id, (template, pattern) in metadata.items():
        by_template[template][example_id.rsplit("-", 1)[0]].add(pattern)
    if len(by_template) != 8:
        raise ValueError(f"Expected eight templates, received {len(by_template)}")
    for template, pairs in by_template.items():
        if len(pairs) != 10 or any(patterns != {"ABBA", "BABA"} for patterns in pairs.values()):
            raise ValueError(f"Expected ten complete ABBA/BABA pairs in {template}")
    return indexed


class TemplateAnalysis:
    """Paired cluster bootstrap: resample whole templates and retain donor dependencies."""

    def __init__(
        self, indexed: Mapping[str, Mapping[str, Row]], *, samples: int, seed: int
    ) -> None:
        if samples < 1:
            raise ValueError("Bootstrap sample count must be positive")
        self.indexed = indexed
        first = next(iter(indexed.values()))
        self.ids = tuple(sorted(first))
        self.templates = tuple(sorted({first[key]["template_id"] for key in self.ids}))
        self.template_indices = [
            np.asarray([i for i, key in enumerate(self.ids) if first[key]["template_id"] == name])
            for name in self.templates
        ]
        self.samples = samples
        self.seed = seed
        self.draws = np.random.default_rng(seed).integers(
            0, len(self.templates), size=(samples, len(self.templates))
        )

    def values(self, spec: str, field: str) -> np.ndarray:
        return np.asarray([self.indexed[spec][key][field] for key in self.ids], dtype=float)

    def gain(self, spec: str, *, logit: bool = False) -> np.ndarray:
        suffix = "_logit" if logit else ""
        return self.values(spec, f"patched{suffix}") - self.values(spec, f"baseline{suffix}")

    def denominator(self, spec: str, *, logit: bool = False) -> np.ndarray:
        suffix = "_logit" if logit else ""
        return self.values(spec, f"clean{suffix}") - self.values(spec, f"baseline{suffix}")

    def estimate(
        self, values: np.ndarray, *, ratios: tuple[tuple[int, int, float, float], ...] = ()
    ) -> dict[str, Any]:
        """Summarize a mean or a signed sum of ratios of whole-template means.

        Each ratio tuple is (numerator_column, denominator_column, sign, threshold).
        A CI is suppressed if any resample has a small/sign-changing denominator;
        unstable bootstrap draws are never silently dropped from a reported CI.
        """
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
        if array.shape[0] != len(self.ids) or not np.isfinite(array).all():
            raise ValueError("Estimate requires one finite, aligned row per example")
        template_means = np.stack(
            [array[indices].mean(axis=0) for indices in self.template_indices]
        )
        point = template_means.mean(axis=0)
        boot = template_means[self.draws].mean(axis=1)
        if not ratios:
            if array.shape[1] != 1:
                raise ValueError("A mean estimate must have one column")
            point_value: float | None = float(point[0])
            boot_values = boot[:, 0]
            template_values = template_means[:, 0].tolist()
            valid = np.ones(self.samples, dtype=bool)
            interval_note = None
        else:
            point_valid = all(abs(point[d]) >= threshold for _, d, _, threshold in ratios)
            valid = np.ones(self.samples, dtype=bool)
            for _, d, _, threshold in ratios:
                valid &= (abs(boot[:, d]) >= threshold) & (boot[:, d] * point[d] > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                boot_values = sum(sign * boot[:, n] / boot[:, d] for n, d, sign, _ in ratios)
            valid &= np.isfinite(boot_values)
            point_value = (
                float(sum(sign * point[n] / point[d] for n, d, sign, _ in ratios))
                if point_valid
                else None
            )
            template_values = []
            for means in template_means:
                if all(abs(means[d]) >= threshold for _, d, _, threshold in ratios):
                    template_values.append(
                        float(sum(sign * means[n] / means[d] for n, d, sign, _ in ratios))
                    )
                else:
                    template_values.append(None)
            interval_note = (
                "Suppressed: point or bootstrap denominator is small or changes sign; "
                "no conditional interval was computed by dropping unstable draws."
                if point_value is None or not valid.all()
                else None
            )
        ci = (
            [float(value) for value in np.percentile(boot_values, (2.5, 97.5))]
            if point_value is not None and valid.all()
            else None
        )
        return {
            "mean": point_value,
            "ci95": ci,
            "per_template": dict(zip(self.templates, template_values, strict=True)),
            "n_examples": len(self.ids),
            "n_templates": len(self.templates),
            "bootstrap_samples": self.samples,
            "bootstrap_seed": self.seed,
            "valid_bootstrap_draws": int(valid.sum()),
            "interval_note": interval_note,
        }

    def outcomes(self, spec: str) -> dict[str, Any]:
        result = {}
        for metric, logit, threshold in (
            ("probability_difference", False, 0.01),
            ("logit_difference", True, 0.1),
        ):
            gain = self.gain(spec, logit=logit)
            denominator = self.denominator(spec, logit=logit)
            result[metric] = {
                "gain": self.estimate(gain),
                "target_clean_effect": self.estimate(denominator),
                "cts": self.estimate(
                    np.column_stack((gain, denominator)), ratios=((0, 1, 1.0, threshold),)
                ),
            }
        return result

    def directional_contrast(self, forward: str, reverse: str) -> dict[str, Any]:
        result: dict[str, Any] = {"forward_spec": forward, "reverse_spec": reverse}
        for metric, logit, threshold in (
            ("probability_difference", False, 0.01),
            ("logit_difference", True, 0.1),
        ):
            forward_gain = self.gain(forward, logit=logit)
            reverse_gain = self.gain(reverse, logit=logit)
            result[metric] = {
                "gain_forward_minus_reverse": self.estimate(forward_gain - reverse_gain),
                "cts_forward_minus_reverse": self.estimate(
                    np.column_stack(
                        (
                            forward_gain,
                            self.denominator(forward, logit=logit),
                            reverse_gain,
                            self.denominator(reverse, logit=logit),
                        )
                    ),
                    ratios=((0, 1, 1.0, threshold), (2, 3, -1.0, threshold)),
                ),
            }
        return result

    def control_contrast(self, correct: str, controls: Sequence[str]) -> dict[str, Any]:
        result: dict[str, Any] = {"correct_spec": correct, "control_specs": list(controls)}
        for metric, logit in (("probability_difference", False), ("logit_difference", True)):
            correct_gain = self.gain(correct, logit=logit)
            control_gains = np.stack([self.gain(spec, logit=logit) for spec in controls])
            mean_control_gain = control_gains.mean(axis=0)
            draw_contrasts = (correct_gain[None, :] - control_gains).mean(axis=1)
            result[metric] = {
                "control_gain": self.estimate(mean_control_gain),
                "correct_minus_control_gain": self.estimate(correct_gain - mean_control_gain),
                "individual_draw_mean_contrasts": dict(
                    zip(controls, [float(value) for value in draw_contrasts], strict=True)
                ),
                "individual_draw_mean_contrast_range": [
                    float(draw_contrasts.min()),
                    float(draw_contrasts.max()),
                ],
            }
        return result

    def blocking(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "interpretation": (
                "Controlled END-node blocking dependence diagnostic, not an isolated "
                "edge path or natural indirect effect. Baseline clamps should be no-ops."
            )
        }
        original = "native_s2_canonical_to_controlled"
        baseline_specs = ["block_selected_baseline"] + [
            f"block_random_{draw}_baseline" for draw in DRAWS
        ]
        for metric, logit in (("probability_difference", False), ("logit_difference", True)):
            original_gain = self.gain(original, logit=logit)
            selected_residual = self.gain("block_selected_s2", logit=logit)
            selected_baseline_gain = self.gain("block_selected_baseline", logit=logit)
            random_residuals = np.stack(
                [self.gain(f"block_random_{draw}_s2", logit=logit) for draw in DRAWS]
            )
            mean_random_residual = random_residuals.mean(axis=0)
            baseline_errors = {
                spec: {
                    "max_absolute": float(np.abs(self.gain(spec, logit=logit)).max()),
                    "mean_absolute": float(np.abs(self.gain(spec, logit=logit)).mean()),
                }
                for spec in baseline_specs
            }
            result[metric] = {
                "original_s2_gain": self.estimate(original_gain),
                "selected_attenuation": self.estimate(original_gain - selected_residual),
                "selected_attenuation_baseline_adjusted": self.estimate(
                    original_gain - selected_residual + selected_baseline_gain
                ),
                "selected_residual_gain": self.estimate(selected_residual),
                "mean_random_attenuation": self.estimate(original_gain - mean_random_residual),
                "selected_minus_mean_random_attenuation": self.estimate(
                    mean_random_residual - selected_residual
                ),
                "individual_random_mean_attenuations": [
                    float((original_gain - residual).mean()) for residual in random_residuals
                ],
                "baseline_clamp_errors": baseline_errors,
                "maximum_baseline_clamp_error": max(
                    value["max_absolute"] for value in baseline_errors.values()
                ),
            }
        return result


def analyze_validated(
    indexed: Mapping[str, Mapping[str, Row]],
    *,
    model: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    analysis = TemplateAnalysis(indexed, samples=samples, seed=seed)
    native = {spec: analysis.outcomes(spec) for spec in indexed if spec.startswith("native_")}
    matched = {spec: analysis.outcomes(spec) for spec in indexed if spec.startswith("matched_")}
    directional = {}
    controls = {}
    mismatch_vectors = []
    for site in SITES:
        forward = f"native_{site}_canonical_to_controlled"
        reverse = f"native_{site}_controlled_to_canonical"
        directional[f"native_{site}"] = analysis.directional_contrast(forward, reverse)
        for dose in DOSES:
            directional[f"matched_{site}_alpha_{dose}"] = analysis.directional_contrast(
                f"matched_{site}_canonical_to_controlled_alpha_{dose}",
                f"matched_{site}_controlled_to_canonical_alpha_{dose}",
            )
        controls[site] = {}
        for control in ("mismatch", "off_target", "random_heads", "random_direction"):
            specs = (
                [f"control_{site}_{control}_{draw}" for draw in DRAWS]
                if control.startswith("random_")
                else [f"control_{site}_{control}"]
            )
            controls[site][control] = analysis.control_contrast(forward, specs)
        mismatch_vectors.extend((analysis.gain(f"control_{site}_mismatch"), analysis.gain(forward)))
    mismatch_array = np.column_stack(mismatch_vectors)
    mismatch = {
        "definition": "Ratio of mean mismatched-donor gain to mean correct gain; not CTS.",
        "s2_retention": analysis.estimate(mismatch_array[:, :2], ratios=((0, 1, 1.0, 1e-8),)),
        "end_retention": analysis.estimate(mismatch_array[:, 2:], ratios=((0, 1, 1.0, 1e-8),)),
        "s2_minus_end_retention": analysis.estimate(
            mismatch_array, ratios=((0, 1, 1.0, 1e-8), (2, 3, -1.0, 1e-8))
        ),
        "end_minus_s2_correct_minus_mismatch_gain": analysis.estimate(
            mismatch_array[:, 3]
            - mismatch_array[:, 2]
            - (mismatch_array[:, 1] - mismatch_array[:, 0])
        ),
    }
    return {
        "model": model,
        "n_examples": len(analysis.ids),
        "n_templates": len(analysis.templates),
        "n_specs": len(indexed),
        "n_rows": sum(len(group) for group in indexed.values()),
        "template_ids": list(analysis.templates),
        "native": native,
        "matched": matched,
        "directional_contrasts": directional,
        "controls": controls,
        "mismatch_stage_comparison": mismatch,
        "blocking": analysis.blocking(),
    }


def analyze_rows(
    rows: Sequence[Row],
    *,
    model: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    return analyze_validated(
        validate_rows(rows, model=model), model=model, samples=samples, seed=seed
    )


def format_estimate(estimate: Mapping[str, Any]) -> str:
    if estimate["mean"] is None:
        return "suppressed"
    point = f"{estimate['mean']:.4f}"
    if estimate["ci95"] is None:
        return f"{point} [CI suppressed]"
    return f"{point} [{estimate['ci95'][0]:.4f}, {estimate['ci95'][1]:.4f}]"


def markdown_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# Study 5: post-hoc robustness analysis",
        "",
        (
            "This extension reuses the eight previously examined test templates. It is descriptive "
            "post-hoc robustness evidence, not a new untouched confirmatory study. No confirmatory "
            "p-values or pass/fail gates are computed."
        ),
        "",
        (
            "Intervals use 5,000 whole-template bootstrap draws (seed 20260905), preserving "
            "all entity pairs and donor links inside each selected template. Five random-control "
            "draws are averaged within each example before inference. All templates have equal "
            "weight. Intervals are pointwise, exploratory, and unadjusted for multiplicity; "
            "they condition on the frozen selected head groups. The small number of templates "
            "limits uncertainty estimates."
        ),
        "",
        (
            "Raw answer-score gains are primary. CTS divides a gain by the target's mean "
            "clean-minus-corrupt effect; it is not causal influence independent of target scaling. "
            "Ratios use ratios of means, never means of per-example ratios. Normalized CIs are "
            "suppressed if any bootstrap denominator is too small or changes sign."
        ),
        "",
        (
            "Matched-dose runs use common head sets and symmetrically matched group-delta norms. "
            "This equalizes total injected norm, not downstream gain or causal efficacy. "
            "Native runs retain the original source-specific groups."
        ),
        "",
        (
            "Historical native/baseline agreement is checked against the original GPU outputs "
            "using the documented CPU numerical tolerances: 0.001 probability-difference and "
            "0.01 logit-difference units. This is bounded numerical agreement, not byte-identical "
            "reproduction. No-op and norm checks retain their separate stricter tolerances."
        ),
        "",
    ]
    for model, values in result["models"].items():
        lines.extend(
            [
                f"## {model}",
                "",
                (
                    f"Validated {values['n_rows']:,} rows: {values['n_specs']} specs × "
                    f"{values['n_examples']} examples across {values['n_templates']} templates."
                ),
                "",
                "### Directionality",
                "",
                (
                    "Differences are canonical→controlled minus controlled→canonical. "
                    "All table entries concern the probability-difference endpoint; the JSON "
                    "also contains logit-difference estimates and each underlying gain/denominator."
                ),
                "",
                "| Intervention | Raw gain difference [95% CI] | CTS difference [95% CI] |",
                "| --- | --- | --- |",
            ]
        )
        for key, contrast in values["directional_contrasts"].items():
            metric = contrast["probability_difference"]
            lines.append(
                f"| {key} | {format_estimate(metric['gain_forward_minus_reverse'])} | "
                f"{format_estimate(metric['cts_forward_minus_reverse'])} |"
            )
        lines.extend(
            [
                "",
                "### Corrected controls",
                "",
                (
                    "Positive contrasts mean the ordinary canonical→controlled patch has a larger "
                    "answer-score gain than the stated control. Draw ranges are descriptive, "
                    "not additional confidence intervals."
                ),
                "",
                "| Site | Control | Correct minus control [95% CI] | Individual-draw mean range |",
                "| --- | --- | --- | --- |",
            ]
        )
        for site, site_controls in values["controls"].items():
            for name, contrast in site_controls.items():
                metric = contrast["probability_difference"]
                bounds = metric["individual_draw_mean_contrast_range"]
                lines.append(
                    f"| {site} | {name} | "
                    f"{format_estimate(metric['correct_minus_control_gain'])} | "
                    f"[{bounds[0]:.4f}, {bounds[1]:.4f}] |"
                )
        mismatch = values["mismatch_stage_comparison"]
        lines.extend(
            [
                "",
                (
                    f"Mismatch retention at S2: {format_estimate(mismatch['s2_retention'])}; "
                    f"at END: {format_estimate(mismatch['end_retention'])}. "
                    f"S2 minus END: {format_estimate(mismatch['s2_minus_end_retention'])}."
                ),
                "",
                "### END-node blocking after an S2 patch",
                "",
                values["blocking"]["interpretation"],
                "",
            ]
        )
        block = values["blocking"]["probability_difference"]
        for key, label in (
            ("original_s2_gain", "Original S2 gain"),
            ("selected_attenuation", "Attenuation from selected END-node clamp"),
            ("selected_residual_gain", "Residual gain under selected END-node clamp"),
            ("mean_random_attenuation", "Attenuation from mean random END-node clamp"),
            ("selected_minus_mean_random_attenuation", "Selected-minus-random attenuation"),
        ):
            lines.append(f"- {label}: {format_estimate(block[key])}.")
        lines.extend(
            [
                "",
                (
                    "Maximum baseline-clamp no-op error: "
                    f"{block['maximum_baseline_clamp_error']:.3g} probability-difference units; "
                    f"{values['blocking']['logit_difference']['maximum_baseline_clamp_error']:.3g} "
                    "logit-difference units."
                ),
                "",
                "### Individual-template diagnostics",
                "",
                (
                    "Each entry is a raw probability-gain contrast. The JSON retains per-template "
                    "statistics for every reported estimate."
                ),
                "",
                (
                    "| Template | Native END forward−reverse | Matched END α=1 forward−reverse | "
                    "Selected−random S2-block attenuation |"
                ),
                "| --- | --- | --- | --- |",
            ]
        )
        native = values["directional_contrasts"]["native_end"]["probability_difference"]
        matched = values["directional_contrasts"]["matched_end_alpha_1"]["probability_difference"]
        for template in values["template_ids"]:
            lines.append(
                f"| {template} | "
                f"{native['gain_forward_minus_reverse']['per_template'][template]:.4f} | "
                f"{matched['gain_forward_minus_reverse']['per_template'][template]:.4f} | "
                f"{block['selected_minus_mean_random_attenuation']['per_template'][template]:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Scope",
            "",
            (
                "These measurements do not identify an edge-isolated pathway, prove algorithmic "
                "equivalence, establish a complete IOI circuit, or show that transport predicts "
                "faithfulness better than overlap/representation baselines. The former raw END "
                "unembedding projection is not used as an independent pathway gate here."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    from verify_experiment_005 import verify_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError("Models must not be repeated")
    validated = {}
    input_hashes = {}
    verification = {}
    for model in args.models:
        path = args.input_root / model / "patches-per-example.jsonl"
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        validated[model] = validate_rows(rows, model=model)
        verification[model] = verify_run(
            args.input_root / model, model=model, indexed=validated[model]
        )
        input_hashes[model] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if len(validated) > 1:
        first_model = next(iter(validated))
        first_rows = next(iter(validated[first_model].values()))
        first_ids = {(key, row["template_id"], row["pattern"]) for key, row in first_rows.items()}
        for indexed in validated.values():
            comparison = next(iter(indexed.values()))
            if {
                (key, row["template_id"], row["pattern"]) for key, row in comparison.items()
            } != first_ids:
                raise ValueError(
                    "Model datasets do not contain identical text/example/template IDs"
                )
    result = {
        "study": "005",
        "status": "completed descriptive post-hoc robustness analysis",
        "data_status": "Previously examined test templates reused; not new confirmatory data.",
        "inference": {
            "unit": "whole template",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "random_draw_handling": "Average five draws per example before template resampling.",
            "normalized_interval_policy": (
                "Suppress the entire interval if any bootstrap denominator is too small or "
                "changes sign relative to the point denominator; do not drop unstable draws."
            ),
            "confirmatory_pvalues": False,
        },
        "input_files": input_hashes,
        "run_verification": verification,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "models": {
            model: analyze_validated(indexed, model=model) for model, indexed in validated.items()
        },
    }
    json_text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown = markdown_report(result)
    # Validate every model and finish all calculations before creating/writing outputs.
    destination = args.output_dir or args.input_root
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "analysis.json").write_text(json_text, encoding="utf-8")
    (destination / "ANALYSIS.md").write_text(markdown, encoding="utf-8")
    print(f"Wrote descriptive Study 5 analysis for {len(validated)} model(s) to {destination}")


if __name__ == "__main__":
    main()
