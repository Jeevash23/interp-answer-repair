"""Draw the confirmation and source-coverage comparison from verified outputs.

Bridge 007 pooled intervals use its sixteen-quantity primary family; Bridge 008
uses its four-quantity family. Cellwise intervals are pointwise in both studies.
This helper only formats saved estimates: no fitting or inference occurs here.
"""
import argparse
import hashlib
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8985'
CELLS = ['w0_o0', 'w0_o1', 'w1_o0', 'w1_o1']
REFERENCES = [('zero', BLUE, .12, 'o', 'Zero replacement'),
              ('fit_mean', ORANGE, -.12, 'D', 'Mean replacement')]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_verified(number):
    """Bind the exact plotted summary and verifier code to the canonical audit."""
    study = ROOT / f'studies/bridge-{number}'
    audit_path = study / 'results/verification.json'
    if not audit_path.exists():
        return None, dict(status='not_ready', reason='Canonical independent verification is not available')
    audit = read(audit_path)
    assert audit['status'] == 'passed'
    assert digest(study / f'code/verify{number}.py') == audit['helper_sha256'], 'Verifier code changed after audit'
    if number == '007':
        status = read(study / 'results/status.json')
        assert status['status'] in ('completed', 'inference_completed')
        assert status['outcome'] == audit['outcome'] and audit['stopping_branch_verified']
        if not status['test_measured']:
            assert status['outcome'] in ('no_screen_pair', 'no_selection_pair', 'independent_qualification_failed')
            assert not (study / 'results/analysis/summary.json').exists()
            return None, dict(status='skipped', reason='Verified stop before final transfer', outcome=status['outcome'])
        assert status['outcome'] == 'final_test_completed'
    else:
        assert number == '008'
        assert digest(ROOT / 'studies/bridge-007/code/verify007.py') == audit['computational_dependency_sha256']
        if audit.get('outcome') == 'ineligible_base':
            status_path = study / 'results/status.json'
            assert digest(status_path) == audit['status_sha256']
            status = read(status_path)
            assert status['status'] == 'completed' and status['outcome'] == 'ineligible_base'
            assert audit['stopping_branch_verified'] and not status['test_measured']
            assert not (study / 'results/analysis/summary.json').exists()
            return None, dict(status='skipped', reason='Verified ineligible Bridge 007 base; no coverage outcomes measured',
                              outcome='ineligible_base')
        completion_path = study / 'results/inference-complete.json'
        assert digest(completion_path) == audit['inference_completion_sha256']
        assert read(completion_path)['status'] == 'completed'
    summary_path = study / 'results/analysis/summary.json'
    assert digest(summary_path) == audit['analysis_summary_sha256'], 'Summary changed after audit'
    summary = read(summary_path)
    assert summary['n'] == 3200 and summary['blocks'] == summary['triples'] == 20
    assert summary['primary_multiplicity'] == (16 if number == '007' else 4)
    return summary, dict(status='ready', summary_sha256=digest(summary_path), verification_sha256=digest(audit_path))


def style():
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})


def coordinates(value, adjusted):
    if value is None or not value.get('valid', True):
        return None
    mean = value['mean']
    lo, hi = value['adjusted_ci' if adjusted else 'ci95']
    assert np.isfinite([mean, lo, hi]).all() and lo <= hi
    return mean, lo, hi


def mark_interval(ax, value, y, color, shift, marker, pooled):
    values = coordinates(value, pooled)
    if values is None:
        ax.text(.03, y+shift, 'not evaluable', transform=ax.get_yaxis_transform(),
                color=color, fontsize=7, va='center')
        return None
    mean, lo, hi = values
    ax.plot([lo, hi], [y+shift]*2, color=color, lw=1.15, zorder=3)
    ax.plot(mean, y+shift, marker, color=color, ms=4.5,
            mfc=color if pooled else 'white', mew=1.0, zorder=4)
    return values


def reference_legend(fig, anchor=(.57, -.04)):
    handles = [Line2D([], [], marker=marker, color=color, lw=1.15, markersize=4.5, label=label)
               for _, color, _, marker, label in REFERENCES]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=anchor,
               ncol=2, frameon=False, fontsize=8)


def draw007(summary):
    """Retain the original prospective absolute/full discrepancy figure."""
    labels = ['Pooled (adjusted)'] + [f'F{summary["families"][int(c[1])]} / order {c[-1]}' for c in CELLS]
    ys = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True,
                             gridspec_kw={'wspace': .2})
    panels = [('mean_absolute_axis_difference', 'absolute_axis_difference',
               '(a) Absolute donor-axis difference', r'mean $|S_{wording}-S_{self}|$', .1),
              ('mean_full_signature_distance', 'full_signature_distance',
               '(b) Full signature discrepancy', 'mean normalised distance', .5)]
    for ax, (primary, cell_metric, title, xlabel, bound) in zip(axes, panels):
        max_upper = bound
        for ref, color, shift, marker, _ in REFERENCES:
            values = [summary['primary'][primary][ref]]
            values += [summary['per_cell_heterogeneity'][ref][c][cell_metric] for c in CELLS]
            for j, (y, value) in enumerate(zip(ys, values)):
                values_drawn = mark_interval(ax, value, y, color, shift, marker, j == 0)
                if values_drawn is not None:
                    max_upper = max(max_upper, values_drawn[0], values_drawn[2])
        ax.axvline(bound, color=GREY, linestyle='--', lw=.8)
        ax.axhline(3.5, color='#dddcd7', lw=.7)
        ax.set_xlim(0, max_upper * 1.15)
        ax.set_ylim(-.5, 4.65)
        ax.set_title(title, pad=7)
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', color='#eeede9', lw=.5)
        ax.text(bound, 4.42, f'bound {bound:.2f}', color='#555555', fontsize=7,
                ha='center', va='bottom')
    axes[0].set_yticks(ys, labels)
    reference_legend(fig)
    return fig


def coverage_values(summary, variant, ref):
    cells = summary['primary_cellwise'][variant][ref]
    return [summary['primary'][variant][ref]] + [None if cells is None else cells[c] for c in CELLS]


def draw008(summary):
    """Pooled adjusted and all cellwise signed interactions; no new estimates."""
    labels = ['Pooled (K=4 adjusted)'] + [f'F{summary["families"][int(c[1])]} / order {c[-1]}' for c in CELLS]
    ys = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharex=True, sharey=True,
                             gridspec_kw={'wspace': .18})
    # One observed-data scale across both panels and both references, always
    # including the zero-interaction baseline. No effect threshold is added.
    limits = [0.0]
    for variant in ('broad', 'matched'):
        for ref, *_ in REFERENCES:
            for j, value in enumerate(coverage_values(summary, variant, ref)):
                values = coordinates(value, j == 0)
                if values is not None:
                    limits.extend(values)
    low, high = min(limits), max(limits)
    padding = .09*(high-low) if high > low else .1
    for ax, variant, title in zip(axes, ('broad', 'matched'),
                                  ('(a) Broad: 112 source heads', '(b) Norm-matched broad')):
        available = any(coordinates(value, j == 0) is not None
                        for ref, *_ in REFERENCES
                        for j, value in enumerate(coverage_values(summary, variant, ref)))
        if available:
            for ref, color, shift, marker, _ in REFERENCES:
                for j, (y, value) in enumerate(zip(ys, coverage_values(summary, variant, ref))):
                    mark_interval(ax, value, y, color, shift, marker, j == 0)
        else:
            name = 'Matched' if variant == 'matched' else 'Broad'
            ax.text(.5, .48, name+' contrast not evaluable\nNo configurations were dropped.',
                    transform=ax.transAxes, ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.95, pad=4), zorder=5)
        ax.axvline(0, color='#555555', linestyle='--', lw=.9, zorder=2)
        ax.axhline(3.5, color='#dddcd7', lw=.7)
        ax.set_xlim(low-padding, high+padding)
        ax.set_ylim(-.5, 4.6)
        ax.set_title(title, pad=7)
        ax.set_xlabel('Change in wording-minus-self donor-axis position\n(relative to the narrow 12-head edit)')
        ax.grid(axis='x', color='#eeede9', lw=.5, zorder=0)
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        ax.locator_params(axis='x', nbins=5)
    axes[0].set_yticks(ys, labels)
    reference_legend(fig, anchor=(.57, -.105))
    fig.text(.57, -.235, 'Filled: pooled, K=4 adjusted intervals.  Open: cellwise 95% intervals.',
             ha='center', va='top', fontsize=7.5, color='#444444')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('007', '008', 'all'), default='all')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'paper/figures')
    parser.add_argument('--check-ready', action='store_true', help='Check audited source bindings without writing figures')
    args = parser.parse_args()
    style()
    results = {}
    filenames = {'007': 'prospective-self-wording.pdf', '008': 'source-coverage-interactions.pdf'}
    for number in (('007', '008') if args.study == 'all' else (args.study,)):
        summary, status = load_verified(number)
        results[number] = dict(status, written=False)
        if summary is None or args.check_ready:
            continue
        figure = draw007(summary) if number == '007' else draw008(summary)
        out = args.output_dir / filenames[number]
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out, bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None,
                       'Subject': f"Bridge {number}; audited summary SHA256 {status['summary_sha256']}"})
        plt.close(figure)
        results[number].update(status='written', written=True, path=str(out))
    print(json.dumps(results, sort_keys=True))


if __name__ == '__main__':
    main()
