"""Independent saved-observation verification; no model inference or new estimand.

Added after the scientific input freeze to implement the computational audit
already specified in the protocol. Imports no Bridge 009 analysis/runner code.
Only writes a completion report when the observed stopping branch is complete.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
SEED, DRAWS, K = 2026100409, 50_000, 16
REFERENCES = ("zero", "fit_mean")
FAMILIES = [10, 11]
BANK_FAMILIES = [10, 11, 12, 13]
PAIRS = [FAMILIES]
SPEC = dict(site=7, layers=[7, 8, 9, 10, 11], heads=12, width=768,
            source_heads=[[0, 1], [3, 0], [5, 5], [6, 9]])
EXPECTED_COUNTS = dict(calibration=32, screen=96, fit=512, selection=512,
                       qualification=1152, test=3200)


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode())
    h.update(json.dumps(value.shape).encode())
    h.update(memoryview(value).cast("B"))
    return h.hexdigest()


class Hashes:
    def __init__(self):
        self.cache = {}

    def __call__(self, path):
        path = Path(path).resolve()
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self.cache:
            with path.open("rb") as stream:
                self.cache[key] = hashlib.file_digest(stream, "sha256").hexdigest()
        return self.cache[key]


def selected_rows(candidates, split, families):
    return [dict(r, wording=families.index(r["family"])) for r in candidates
            if r["split"] == split and r["family"] in families]


def masks(rows):
    return {f"w{w}_o{o}": np.array([r["wording"] == w and r["order"] == o for r in rows])
            for w in (0, 1) for o in (0, 1)}


def donors(rows):
    # Construct by direct partner search independently of the analyzer's key map.
    result = []
    for row in rows:
        match = [j for j, other in enumerate(rows)
                 if all(other[key] == row[key] for key in ("template", "pair", "role", "order"))
                 and other["wording"] != row["wording"]]
        assert len(match) == 1, "Nonunique or missing donor partner"
        result.append(match[0])
    result = np.asarray(result)
    assert np.array_equal(result[result], np.arange(len(rows)))
    return result


def behavior(rows, values):
    out = {}
    for family in sorted({r["family"] for r in rows}):
        out[str(family)] = {}
        for order in (0, 1):
            b = values[[r["family"] == family and r["order"] == order for r in rows]]
            out[str(family)][str(order)] = dict(n=len(b), pairwise=float(np.mean(b[:, 2])),
                                                top1=float(np.mean(b[:, 3])), logit=float(np.mean(b[:, 0])))
    return out


def behavioral_gate(cells):
    return all(c["pairwise"] >= .95 and c["top1"] >= .85 for orders in cells.values() for c in orders.values())


def refit(raw, rows):
    observations = raw[..., 0].astype(np.float64)
    grouped = [np.array([2 * r["wording"] + r["order"] == k for r in rows]) for k in range(4)]
    means = np.stack([observations[:, group].mean(1) for group in grouped], axis=1)
    errors = np.empty_like(observations)
    for k, group in enumerate(grouped):
        errors[:, group] = observations[:, group] - means[:, k, None, :]
    scales = np.sqrt((errors * errors).mean(1))
    scales[scales < .1] = .1
    return dict(scale=scales, centers=means / scales[:, None, :])


def classify(raw, rows, ref):
    x = raw[..., 0].astype(float) / np.asarray(ref["scale"])[:, None, :]
    centers = np.asarray(ref["centers"])
    correct = np.empty(x.shape[:2], bool)
    axis_valid = np.ones(x.shape[:2], bool)
    # Nearest-centroid squared distances are independent of the projection implementation.
    for j, row in enumerate(rows):
        a, b = centers[:, row["order"]], centers[:, row["order"] + 2]
        da = np.sum((x[:, j] - a) ** 2, axis=-1)
        db = np.sum((x[:, j] - b) ** 2, axis=-1)
        axis_valid[:, j] = np.sum((b - a) ** 2, axis=-1) >= 1e-12
        correct[:, j] = ((db < da) == bool(row["wording"])) & axis_valid[:, j]
    return correct, axis_valid


def class_rates(raw, rows, ref):
    correct, _ = classify(raw, rows, ref)
    return {cell: correct[:, mask].mean(1).tolist() for cell, mask in masks(rows).items()}


def close_tree(actual, expected, label, atol=1e-11):
    if isinstance(actual, dict):
        assert set(actual) == set(expected), label
        for key in actual:
            close_tree(actual[key], expected[key], label + "/" + str(key), atol)
    elif isinstance(actual, (list, tuple, np.ndarray)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=label)
    elif isinstance(actual, (bool, str)) or actual is None:
        assert actual == expected, label
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=label)


class ExplicitBootstrap:
    """Old-style grid indexing; intentionally not the analyzer's count-weight method."""
    def __init__(self, rows, seed=SEED, draws=DRAWS):
        self.rows = rows
        self.blocks = sorted({r["template"] for r in rows})
        self.triples = sorted({r["pair"] for r in rows})
        rng = np.random.default_rng(seed)
        self.bt = rng.integers(len(self.blocks), size=(draws, len(self.blocks)))
        self.bp = rng.integers(len(self.triples), size=(draws, len(self.triples)))
        self.draws = draws

    def interval(self, values, mask=None, multiplicity=K):
        mask = np.ones(len(self.rows), bool) if mask is None else mask
        values = np.asarray(values, float)
        grid = np.empty((len(self.blocks), len(self.triples)))
        for i, block in enumerate(self.blocks):
            for j, triple in enumerate(self.triples):
                chosen = mask & np.array([r["template"] == block and r["pair"] == triple for r in self.rows])
                assert chosen.any(), "Missing design cell"
                grid[i, j] = np.mean(values[chosen])
        if not np.isfinite(grid).all():
            return dict(mean=None, ci95=None, adjusted_ci=None, valid=False)
        distribution = np.empty(self.draws)
        # Chunked indexing keeps peak memory small without changing any resample.
        for start in range(0, self.draws, 1_000):
            end = min(start + 1_000, self.draws)
            distribution[start:end] = grid[self.bt[start:end, :, None], self.bp[start:end, None, :]].mean((1, 2))
        tail = .05 / (2 * multiplicity)
        return dict(mean=float(grid.mean()), ci95=np.quantile(distribution, [.025, .975]).tolist(),
                    adjusted_ci=np.quantile(distribution, [tail, 1-tail]).tolist(), valid=True)


def audit_freezes(hasher):
    summaries, lookup = {}, {}
    for path in sorted((HERE / "data").glob("*-freeze.json")):
        manifest = read(path)
        assert manifest["files"], path
        for name, value in manifest["files"].items():
            expected = value["sha256"] if isinstance(value, dict) else value
            target = Path(name) if Path(name).is_absolute() else ROOT / name
            assert hasher(target) == expected, str(target)
        sha = hasher(path)
        summaries[path.name] = dict(sha256=sha, verified_files=len(manifest["files"]), utc=manifest.get("utc"))
        lookup[sha] = path
    assert "input-freeze.json" in summaries
    return summaries, lookup


def audit_preparation(candidates, hasher):
    """Recheck the inherited design and tokenize locally, without importing a model."""
    from tokenizers import Tokenizer

    prep = read(HERE / "data/preparation-audit.json")
    design = read(HERE / "data/design.json")
    assert design["study"] == "bridge-009" and design["model"] == "gpt2-small"
    assert design["seed"] == SEED and design["inherited_data_seed"] == 2026100407
    assert design["family_ids"] == FAMILIES and design["bank_family_ids"] == BANK_FAMILIES
    assert design["candidate_pairs"] == design["alignment_eligible_pairs"] == PAIRS
    assert prep["fixed_pair"] == FAMILIES and prep["bank_families"] == BANK_FAMILIES
    assert prep["model_spec"] == read(ROOT / "configs/study.json")["models"]["gpt2-small"] == SPEC
    assert prep["fit_mean_bank_rows"] == 1024 and prep["fit_signature_rows"] == 512
    assert prep["prepare_source_sha256"] == hasher(HERE / "code/prepare009.py")
    for field in ("inherited_source_hashes", "checkpoint_and_tokenizer_hashes"):
        for name, expected in prep[field].items():
            assert hasher(ROOT / name) == expected, (field, name)
    old = ROOT / "studies/bridge-007"
    assert hasher(HERE / "data/candidates.json") == hasher(old / "data/candidates.json")
    assert read(old / "data/selection.json")["families"] == FAMILIES
    # Inherited manifest hashes alone would not verify the original frozen files.
    for name in ("input-freeze.json", "selection-freeze.json"):
        for filename, expected in read(old / "data" / name)["files"].items():
            assert hasher(ROOT / filename) == expected, filename
    tokenizer = Tokenizer.from_file(str(ROOT / "models/gpt2-small/tokenizer.json"))
    assert tokenizer.token_to_id("<|endoftext|>") == 50256
    assert len(candidates) == 11008 and len({r["id"] for r in candidates}) == len(candidates)
    name_sets, context_sets = {}, {}
    counts, selected_counts, alignments = {}, {}, 0
    for split in EXPECTED_COUNTS:
        rows = [r for r in candidates if r["split"] == split]
        blocks, names = design["blocks"][split], design["names"][split]
        name_sets[split], context_sets[split] = set(names), set(map(tuple, blocks))
        assert len(names) == len(set(names)) and len(names) % 3 == 0
        assert len(blocks) == len(context_sets[split])
        keys = [(r["template"], r["pair"], r["role"], r["family"], r["order"]) for r in rows]
        expected_keys = {(t, p, role, f, o) for t in range(len(blocks)) for p in range(len(names)//3)
                         for role in (0, 1) for f in BANK_FAMILIES for o in (0, 1)}
        assert len(keys) == len(set(keys)) and set(keys) == expected_keys
        counts[split] = len(rows)
        selected = selected_rows(candidates, split, FAMILIES)
        selected_counts[split] = len(selected)
        paired = donors(selected)
        for i, r in enumerate(selected):
            other = selected[paired[i]]
            assert len(r["token_ids"]) == len(other["token_ids"]) and r["s2"] == other["s2"]
            assert all(r[k] == other[k] for k in ("io", "subject", "alternative"))
            alignments += r["family"] == 10
        for r in rows:
            place, obj = blocks[r["template"]]
            triple = names[3*r["pair"]:3*r["pair"]+3]
            io, subject, alternative = triple[r["role"]], triple[1-r["role"]], triple[2]
            assert (r["io"], r["subject"], r["alternative"]) == (io, subject, alternative)
            first, second = (io, subject) if r["order"] == 0 else (subject, io)
            expected_text = design["families"][str(r["family"])].format(
                a=first, b=second, s=subject, place=place, obj=obj,
                article="an" if obj[0].lower() in "aeiou" else "a")
            assert r["text"] == expected_text
            assert r["token_ids"] == [50256] + tokenizer.encode(expected_text, add_special_tokens=False).ids
            for name, key in (("io", "io_id"), ("subject", "subject_id"), ("alternative", "alt_id")):
                assert tokenizer.encode(" " + r[name], add_special_tokens=False).ids == [r[key]]
            assert r["token_ids"][r["s2"]] == r["subject_id"]
            assert r["token_ids"][r["io_position"]] == r["io_id"]
            assert r["token_ids"][r["subject_position"]] == r["subject_id"]
            corrupt = list(r["token_ids"])
            corrupt[r["s2"]] = r["alt_id"]
            assert corrupt == r["corrupt_tokens"]
            assert sum(a != b for a, b in zip(corrupt, r["token_ids"])) == 1
    for banks in (name_sets, context_sets):
        for i, first in enumerate(banks):
            assert all(not banks[first] & banks[second] for second in list(banks)[i+1:])
    assert counts == prep["rows_per_split"] == {k: 2*v for k, v in EXPECTED_COUNTS.items()}
    assert selected_counts == prep["selected_pair_rows_per_split"] == EXPECTED_COUNTS
    assert alignments == prep["matched_pair_alignments"] == 2752
    return dict(all_candidate_rows=len(candidates), fixed_pair=FAMILIES, bank_families=BANK_FAMILIES,
                selected_pair_rows_per_split=selected_counts, model_spec=SPEC,
                exact_candidate_bytes_inherited=True, local_small_tokenization_reproduced=True,
                split_names_disjoint=True, split_context_pairs_disjoint=True,
                checkpoint_and_tokenizer_hashes=prep["checkpoint_and_tokenizer_hashes"],
                inherited_manifests_and_files_verified=True)


def audit_amendment(hasher):
    """Preserve and verify the pre-inference correction instead of concealing it."""
    path = HERE / "data/pre-inference-amendment.json"
    change = read(path)
    assert change["kind"] == "Pre-inference metadata correction"
    assert change["numerical_implementation_unchanged"] is True
    assert change["scientific_inference_started"] is False
    assert change["prior_results_directory_existed"] is False
    overrides = change["archived_path_overrides"]
    prefix = "studies/bridge-009/"
    source_name = prefix + "code/secondary009.py"
    input_name = prefix + "data/input-freeze.json"
    secondary_name = prefix + "data/secondary-analysis-freeze.json"
    assert set(overrides) == {source_name, input_name, secondary_name}
    assert set(change["archived_sha256"]) == set(overrides.values())
    for name, expected in change["archived_sha256"].items():
        assert hasher(ROOT / name) == expected, name
    old_text, new_text = change["exact_text_change"]["old"], change["exact_text_change"]["new"]
    assert old_text == "seed 2026100407; no multiplicity claims"
    assert new_text == "seed 2026100409; no multiplicity claims"
    source = (ROOT / source_name).read_text()
    archived_source = (ROOT / overrides[source_name]).read_text()
    assert archived_source.count(old_text) == 1
    assert archived_source.replace(old_text, new_text) == source
    assert hasher(ROOT / source_name) == change["corrected_source_sha256"]
    stamp = datetime.fromisoformat(change["utc"])
    for name in (input_name, secondary_name):
        old_manifest = read(ROOT / overrides[name])
        new_manifest = read(ROOT / name)
        assert datetime.fromisoformat(old_manifest["utc"]) < stamp
        assert stamp < datetime.fromisoformat(new_manifest["utc"])
        for file_name, expected in old_manifest["files"].items():
            assert hasher(ROOT / overrides.get(file_name, file_name)) == expected, file_name
        amended_files = dict(old_manifest["files"])
        amended_files[source_name] = hasher(ROOT / source_name)
        if name == input_name:
            amended_files[str(path.relative_to(ROOT))] = hasher(path)
        else:
            amended_files[input_name] = hasher(ROOT / input_name)
            assert json_hash(old_manifest["planned_outputs"]) == old_manifest["planned_outputs_sha256"]
            assert json_hash(new_manifest["planned_outputs"]) == new_manifest["planned_outputs_sha256"]
            old_plan = json.dumps(old_manifest["planned_outputs"], sort_keys=True)
            assert old_plan.replace(old_text, new_text) == json.dumps(new_manifest["planned_outputs"], sort_keys=True)
        assert amended_files == new_manifest["files"]
    for record in (HERE / "results").rglob("*-record.json"):
        value = read(record)
        if "input_key" in value:
            assert stamp < datetime.fromisoformat(value["started"]), record
    return dict(verified=True, amendment_sha256=hasher(path), exact_seed_label_only=True,
                old_manifests_and_original_files_verified=True,
                old_source_preserved=True, all_saved_measurements_started_after_amendment=True,
                archived_sha256=change["archived_sha256"])


def audit_small_identity(identity, n, aggregate):
    assert identity["model"] == "gpt2-small" and identity["model_spec"] == SPEC
    assert identity["source"] == SPEC["source_heads"] and identity["layers"] == SPEC["layers"]
    assert identity["batch_size"] == 8 and identity["threads"] == 4
    assert identity["head_chunk"] == 8 and identity["dtype"] == "float32"
    expected = {"behavior": (n, 6)}
    if identity["kind"] == "natural":
        expected.update(z=(n, 5, 12, 64), head_results=(n, 4, 768), state=(n, 768))
        assert identity["bank_sha256"] is None and identity["delta_sha256"] is None
    elif identity["kind"] == "signature":
        expected["raw"] = (2, n, 60, 2)
        assert identity["bank_sha256"] is not None
    else:
        assert identity["kind"] == "behavior" and identity["bank_sha256"] is None
    assert set(aggregate) == set(expected)
    for key, shape in expected.items():
        assert aggregate[key].shape == shape and aggregate[key].dtype == np.float32, (key, shape)
    return expected


def float32_delta(rows, base_delta, factor):
    """Reconstruct the prescribed donor transformation without importing its code."""
    assert base_delta.dtype == np.float32 and base_delta.shape[0] == len(rows)
    assert factor in ("self", "wording", "zero", "order", "role", "order_role")
    keys = [(r["template"], r["pair"], r["role"], r["wording"], r["order"]) for r in rows]
    index = {key: i for i, key in enumerate(keys)}
    assert len(index) == len(rows)
    chosen = []
    for t, p, role, wording, order in keys:
        target = (t, p, 1-role if factor in ("role", "order_role") else role,
                  1-wording if factor in ("wording", "zero") else wording,
                  1-order if factor in ("order", "order_role") else order)
        assert target in index, (factor, target)
        chosen.append(index[target])
    result = base_delta[np.asarray(chosen)]
    if factor == "zero":
        # Multiplication preserves signed zero exactly as in the measurement pipeline.
        result = result * np.float32(0)
    return result


def delta_source(folder, candidates_by_id, calibration=False):
    names = ("edit-clean", "edit-corrupt") if calibration else ("natural-clean", "natural-corrupt")
    clean_record, corrupt_record = (read(folder / (name + "-record.json")) for name in names)
    assert clean_record["ids"] == corrupt_record["ids"], "Mismatched natural source ordering"
    ids = clean_record["ids"]
    family_ids = sorted({candidates_by_id[key]["family"] for key in ids})
    rows = [dict(candidates_by_id[key], wording=family_ids.index(candidates_by_id[key]["family"])) for key in ids]
    with np.load(folder / (names[0] + ".npz"), allow_pickle=False) as c:
        clean_heads = c["head_results"]
    with np.load(folder / (names[1] + ".npz"), allow_pickle=False) as c:
        corrupt_heads = c["head_results"]
    assert clean_heads.dtype == corrupt_heads.dtype == np.float32
    base = clean_heads - corrupt_heads
    assert base.dtype == np.float32 and np.isfinite(base).all()
    return dict(ids=ids, rows=rows, base=base,
                source_aggregate_hashes={names[0]: clean_record["aggregate_sha256"],
                                         names[1]: corrupt_record["aggregate_sha256"]})


def verify_delta_record(record_path, record, rr, by_id, cache):
    identity = record["inputs"]
    expected = identity["delta_sha256"]
    if expected is None:
        return dict(status="no_delta_supplied")
    stem = record_path.name.removesuffix("-record.json")
    if stem.startswith(("residual", "projected")):
        return dict(status="record_hash_bound_only", reason="Torch Gaussian draws not independently reproduced")
    is_calibration = record_path.parent.name == "calibration"
    if is_calibration:
        folder, factor = record_path.parent, "wording"
    else:
        assert record_path.parent.name in ("test", "verification"), record_path
        folder = HERE / "results/test"
        factor = "zero" if record_path.parent.name == "verification" and stem == "corrupt" else stem
    assert factor in ("wording", "self", "zero", "order", "role", "order_role"), (record_path, factor)
    key = str(folder)
    if key not in cache:
        cache[key] = delta_source(folder, by_id, calibration=is_calibration)
    source = cache[key]
    # Compute using the entire source factorial grid, then take requested rows.
    # This also supports manifest-selected verification subsets.
    reconstructed = float32_delta(source["rows"], source["base"], factor)
    position = {key: i for i, key in enumerate(source["ids"])}
    selected = reconstructed[np.array([position[r["id"]] for r in rr])]
    actual = array_hash(selected)
    assert actual == expected, f"Donor delta mismatch: {record_path}"
    return dict(status="independently_reconstructed", factor=factor, delta_sha256=actual,
                source_aggregate_hashes=source["source_aggregate_hashes"])


def audit_runner_records(candidates, hasher, freezes):
    by_id = {r["id"]: r for r in candidates}
    out = []
    source_cache = {}
    for record_path in sorted((HERE / "results").rglob("*-record.json")):
        record = read(record_path)
        if "input_key" not in record:
            continue
        assert record["status"] == "completed", f"Incomplete runner record: {record_path}"
        identity = record["inputs"]
        assert json_hash(identity) == record["input_key"], record_path
        freeze = freezes[identity["freeze_sha256"]]
        phase = record_path.parent.name
        expected_freeze = {"calibration": "input-freeze.json", "results": "input-freeze.json",
                           "fit": "screen-freeze.json", "selection": "screen-freeze.json",
                           "qualification": "selection-freeze.json", "test": "test-freeze.json",
                           "verification": "test-freeze.json"}[phase]
        assert freeze.name == expected_freeze, record_path
        stamp = read(freeze).get("utc")
        if stamp:
            assert datetime.fromisoformat(stamp) <= datetime.fromisoformat(record["started"]), record_path
        assert datetime.fromisoformat(record["started"]) <= datetime.fromisoformat(record["finished"])
        for name, sha in identity["source_sha256"].items():
            assert hasher(ROOT / name) == sha, name
        families = sorted({by_id[key]["family"] for key in record["ids"]})
        rr = [dict(by_id[key], wording=families.index(by_id[key]["family"])) for key in record["ids"]]
        if json_hash(rr) != identity["rows_sha256"]:
            rr = [dict(r, token_ids=r["corrupt_tokens"]) for r in rr]
        assert json_hash(rr) == identity["rows_sha256"], f"Rows changed: {record_path}"
        assert len(rr) == identity["n"] and len({r["id"] for r in rr}) == len(rr)
        stem = record_path.name.removesuffix("-record.json")
        aggregate_path = record_path.with_name(stem + ".npz")
        assert hasher(aggregate_path) == record["aggregate_sha256"], aggregate_path
        aggregate = arrays(aggregate_path)
        audit_small_identity(identity, len(rr), aggregate)
        expected_config = dict(read(ROOT / "configs/study.json"), threads=4)
        assert identity["configuration"] == expected_config
        expected_chunks = []
        for length in sorted({len(r["token_ids"]) for r in rr}):
            indices = [i for i, r in enumerate(rr) if len(r["token_ids"]) == length]
            expected_chunks.extend(np.array(indices[start:start+identity["batch_size"]])
                                   for start in range(0, len(indices), identity["batch_size"]))
        assert record["completed_chunks"] == len(expected_chunks) == len(record["hashes"])
        found = set()
        for i, indices in enumerate(expected_chunks):
            name = f"{stem}-chunk{i:04d}.npz"
            path = record_path.with_name(name)
            assert hasher(path) == record["hashes"][name], path
            chunk = arrays(path)
            assert str(chunk.pop("_resume_input_key").item()) == record["input_key"], path
            np.testing.assert_array_equal(chunk.pop("_resume_indices"), indices)
            assert set(chunk) == set(aggregate)
            for key, value in chunk.items():
                expected = aggregate[key][:, indices] if key == "raw" else aggregate[key][indices]
                np.testing.assert_array_equal(value, expected, err_msg=f"{path}:{key}")
                assert np.isfinite(value).all(), path
            found.update(indices.tolist())
        assert found == set(range(len(rr)))
        # Bank binding is reconstructible without rerunning any model.
        if identity["bank_sha256"] is not None:
            bank = arrays(HERE / "results/fit/bank.npz")["bank"]
            assert bank.shape == (5, 12, 64) and bank.dtype == np.float32
            assert array_hash(bank) == identity["bank_sha256"], record_path
        delta_check = verify_delta_record(record_path, record, rr, by_id, source_cache)
        out.append(dict(path=str(record_path.relative_to(HERE)), chunks=len(expected_chunks), n=len(rr),
                        aggregate_sha256=record["aggregate_sha256"], rows_verified=True,
                        delta_verification=delta_check))
    assert out, "No completed runner records"
    return out


def audit_screen(candidates):
    rows = selected_rows(candidates, "screen", FAMILIES)
    saved = read(HERE / "results/screen.json")
    derived = behavior(rows, arrays(HERE / "results/screen.npz")["behavior"])
    close_tree(derived, saved["cells"], "screen behavior")
    eligible = [int(f) for f, cells in derived.items() if behavioral_gate({f: cells})]
    alignment = read(HERE / "data/design.json")["alignment_eligible_pairs"]
    pairs = [pair for pair in PAIRS if pair in alignment and all(f in eligible for f in pair)]
    assert eligible == saved["eligible"] and pairs == saved["pairs"]
    return saved


def audit_record_coverage(records, outcome):
    """Every prescribed completed measurement needs its hash-bound record."""
    expected = {"calibration/natural", "calibration/edit-clean", "calibration/edit-corrupt", "screen"}
    if outcome != "no_screen_pair":
        expected.update({"fit/natural", "fit/clean", "selection/clean"})
    if outcome not in ("no_screen_pair", "no_selection_pair"):
        expected.add("qualification/clean")
    if outcome == "final_test_completed":
        conditions = ["natural-clean", "natural-corrupt", "clean", "corrupt", "wording", "self",
                      "zero", "order", "role", "order_role"]
        conditions += [kind + str(i) for kind in ("residual", "projected") for i in range(3)]
        expected.update("test/" + c for c in conditions)
        expected.update(("verification/wording", "verification/corrupt"))
    actual = {r["path"].removeprefix("results/").removesuffix("-record.json") for r in records}
    assert actual == expected, {"missing": sorted(expected-actual), "unexpected": sorted(actual-expected)}
    return dict(expected_records=len(expected), exact_measurement_set=True)


def audit_selection(candidates, screen):
    saved = read(HERE / "data/selection.json")
    assert screen["pairs"] == PAIRS
    assert saved["fixed_pair_from"] == "bridge-007" and saved["no_model_specific_pair_search"] is True
    assert saved["bank_families"] == BANK_FAMILIES and saved["source_heads"] == SPEC["source_heads"]
    train = selected_rows(candidates, "fit", FAMILIES)
    valid = selected_rows(candidates, "selection", FAMILIES)
    bank_rows = selected_rows(candidates, "fit", BANK_FAMILIES)
    assert len(train) == len(valid) == 512 and len(bank_rows) == 1024
    a = arrays(HERE / "results/fit/clean.npz")
    b = arrays(HERE / "results/selection/clean.npz")
    nat = arrays(HERE / "results/fit/natural.npz")
    np.testing.assert_array_equal(nat["z"].mean(0), arrays(HERE / "results/fit/bank.npz")["bank"])
    assert nat["z"].shape == (1024, 5, 12, 64)
    bank_indices = {r["id"]: i for i, r in enumerate(bank_rows)}
    selected_indices = np.array([bank_indices[r["id"]] for r in train])
    np.testing.assert_allclose(nat["behavior"][selected_indices], a["behavior"], rtol=0, atol=.0005)
    derived = {}
    for pair in screen["pairs"]:
        ii = np.array([i for i, r in enumerate(train) if r["family"] in pair])
        jj = np.array([i for i, r in enumerate(valid) if r["family"] in pair])
        rr = [dict(train[i], wording=pair.index(train[i]["family"])) for i in ii]
        vv = [dict(valid[i], wording=pair.index(valid[i]["family"])) for i in jj]
        ref = refit(a["raw"][:, ii], rr)
        tc, vc = behavior(rr, a["behavior"][ii]), behavior(vv, b["behavior"][jj])
        ts, vs = class_rates(a["raw"][:, ii], rr, ref), class_rates(b["raw"][:, jj], vv, ref)
        worst, average = float(np.min(list(vs.values()))), float(np.mean(list(vs.values())))
        record = dict(families=pair, reference={k: v.tolist() for k, v in ref.items()}, fit_behavior=tc,
                      selection_behavior=vc, fit_separation=ts, selection_separation=vs, worst=worst,
                      average=average, eligible=bool(behavioral_gate(tc) and behavioral_gate(vc)
                                                    and np.min(list(ts.values())) >= .8 and worst >= .9))
        derived[",".join(map(str, pair))] = record
    close_tree(derived, saved["scores"], "selection")
    ranked = sorted([v for v in derived.values() if v["eligible"]], key=lambda v: (-v["worst"], -v["average"], v["families"]))
    assert saved["families"] == (ranked[0]["families"] if ranked else None)
    assert saved["status"] == ("selected" if ranked else "no_qualified_pair")
    if ranked:
        close_tree(ranked[0]["reference"], saved["reference"], "selected reference")
    return saved, dict(candidates=len(derived), eligible_pairs=len(ranked), families=saved["families"],
                        scores={key: {k: v[k] for k in ("worst", "average", "eligible")} for key, v in derived.items()},
                        bank_families=BANK_FAMILIES, mean_bank_rows=1024, fit_signature_rows=512,
                        no_model_specific_pair_search=True)


def audit_qualification(rows, raw, b, ref, saved):
    correct, axes_valid = classify(raw, rows, ref)
    boot = ExplicitBootstrap(rows)
    maximum_error = 0.0
    for cell, mask in masks(rows).items():
        for key, column in (("pairwise", 2), ("top1", 3)):
            value = boot.interval(b[:, column], mask, multiplicity=1)
            close_tree(value, {k: saved["cells"][cell][key][k] for k in value}, f"qualification/{cell}/{key}")
        for i, ref_name in enumerate(REFERENCES):
            value = boot.interval(correct[i].astype(float), mask, multiplicity=1)
            recorded = saved["cells"][cell]["classification"][ref_name]
            close_tree(value, {k: recorded[k] for k in value}, f"qualification/{cell}/{ref_name}")
            maximum_error = max(maximum_error, abs(value["mean"] - recorded["mean"]))
    r = raw[..., 0].astype(float)
    paired_norm = np.sqrt(np.sum(((r[:, donors(rows)] - r) / np.asarray(ref["scale"])[:, None]) ** 2, axis=-1))
    unresolvable = int(np.sum(paired_norm < 1e-6))
    sp = bool(axes_valid.all() and np.min([correct[:, mask].mean(1) for mask in masks(rows).values()]) >= .8)
    bp = behavioral_gate(behavior(rows, b))
    assert saved["behavior_pass"] == bp and saved["separation_pass"] == sp
    assert saved["qualified"] == bool(bp and sp and unresolvable == 0)
    assert saved["unresolvable_reference_examples"] == unresolvable
    for i, name in enumerate(REFERENCES):
        np.testing.assert_allclose(saved["pair_norm_minimum"][name], paired_norm[i].min(), rtol=0, atol=1e-11)
    return dict(qualified=saved["qualified"], full_pointwise_intervals_reproduced=True,
                maximum_rate_error=maximum_error, unresolvable_reference_examples=unresolvable)


def primary_values(rows, data, reference):
    r = data["clean"]["raw"][..., 0].astype(float)
    d = r[:, donors(rows)]
    scale = np.asarray(reference["scale"])[:, None, :]
    axis = (d-r) / scale
    norm = np.sqrt(np.sum(axis * axis, axis=-1))
    safe = np.where(norm >= 1e-6, norm, np.nan)
    ew = data["wording"]["raw"][..., 0].astype(float)
    es = data["self"]["raw"][..., 0].astype(float)
    # Work directly from the difference of edited states, without computing either S.
    difference = (ew-es) / scale
    absolute_axis = np.abs(np.sum(difference * axis, axis=-1) / safe**2)
    discrepancy = np.sqrt(np.sum(difference**2, axis=-1)) / safe
    margin = (np.sqrt(np.sum(((ew-r)/scale)**2, axis=-1))
              - np.sqrt(np.sum(((ew-d)/scale)**2, axis=-1))) / safe
    gain = data["wording"]["behavior"][:, 0] - data["corrupt"]["behavior"][:, 0]
    outputs = {}
    for cell, mask in masks(rows).items():
        outputs[("recovery", cell)] = (gain, mask)
    for i, name in enumerate(REFERENCES):
        for cell, mask in masks(rows).items():
            outputs[("margin", name, cell)] = (margin[i], mask)
        outputs[("mean_absolute_axis_difference", name)] = (absolute_axis[i], None)
        outputs[("mean_full_signature_distance", name)] = (discrepancy[i], None)
    assert len(outputs) == 16
    return outputs


def audit_primary(rows, data, ref, summary, seed=SEED, multiplicity=K, draws=DRAWS, original_only=False):
    values = primary_values(rows, data, ref)
    boot = ExplicitBootstrap(rows, seed, draws)
    results, max_mean, max_interval = {}, 0., 0.
    for key, (numbers, mask) in values.items():
        if original_only and key[0] not in ("recovery", "margin"):
            continue
        recorded = summary["primary"]
        for part in key:
            recorded = recorded[part]
        estimate = boot.interval(numbers, mask, multiplicity)
        assert recorded.get("valid", True) == estimate["valid"], "/".join(key)
        for field, expected in (("draws", draws), ("seed", seed), ("multiplicity", multiplicity),
                                ("blocks", len(boot.blocks)), ("triples", len(boot.triples))):
            if field in recorded:
                assert recorded[field] == expected, (key, field)
        for field in ("mean", "ci95", "adjusted_ci"):
            close_tree(estimate[field], recorded[field], "/".join(key) + "/" + field, atol=2e-10)
        if estimate["valid"]:
            max_mean = max(max_mean, abs(estimate["mean"]-recorded["mean"]))
            max_interval = max(max_interval, float(np.max(np.abs(np.asarray(estimate["adjusted_ci"])-recorded["adjusted_ci"]))))
        results["/".join(key)] = estimate
    return dict(endpoints=len(results), draws=draws, seed=seed, multiplicity=multiplicity,
                max_point_error=max_mean, max_adjusted_interval_error=max_interval, reproduced=results)


def audit_final_decisions(summary, reproduced):
    def supported(prefix, side, threshold):
        chosen = [v for key, v in reproduced.items() if key.startswith(prefix + "/")]
        assert chosen
        return all(v["valid"] and (v["adjusted_ci"][0] > threshold if side == "above"
                                  else v["adjusted_ci"][1] < threshold) for v in chosen)
    outcomes = dict(all_recovery_intervals_positive=supported("recovery", "above", 0),
                    all_margin_intervals_recipient=supported("margin", "below", 0),
                    average_absolute_axis_bound_supported=supported("mean_absolute_axis_difference", "below", .10),
                    average_full_distance_bound_supported=supported("mean_full_signature_distance", "below", .50))
    close_tree(outcomes, summary["outcomes"], "primary decisions")
    qualified = bool(summary["qualification"]["qualified"] and summary["test_qualification"]["qualified"]
                     and summary["unresolvable_reference_examples"] == 0)
    assert summary["qualified"] == qualified
    assert summary["full_conjunction_supported"] == bool(qualified and all(outcomes.values()))
    return dict(qualified=qualified, outcomes=outcomes, full_conjunction_supported=summary["full_conjunction_supported"])


def self_test():
    # Known factorial ordering supplies a direct row-index oracle for every donor factor.
    fixture_rows = [dict(id=str(i), template=0, pair=0, role=role, wording=w, order=o)
                    for i, (role, w, o) in enumerate((role, w, o) for role in (0, 1) for w in (0, 1) for o in (0, 1))]
    base = np.arange(48, dtype=np.float32).reshape(8, 2, 3) - np.float32(20)
    for factor, flip in (("self", 0), ("wording", 2), ("order", 1), ("role", 4), ("order_role", 5)):
        expected = base[np.arange(8) ^ flip]
        result = float32_delta(fixture_rows, base, factor)
        np.testing.assert_array_equal(result, expected)
        assert array_hash(result) == array_hash(expected)
    zero = float32_delta(fixture_rows, base, "zero")
    assert array_hash(zero) == array_hash(base[np.arange(8) ^ 2] * np.float32(0))
    assert np.signbit(zero).any(), "Signed-zero preservation not exercised"
    # The calibration subset has both wordings/orders but only role 0.
    small = float32_delta(fixture_rows[:4], base[:4], "wording")
    np.testing.assert_array_equal(small, base[[2, 3, 0, 1]])
    # A 60-coordinate known-answer experiment exercises all 16 Small endpoints.
    synthetic_rows = [dict(template=t, pair=p, role=role, wording=w, order=o, family=10+w)
                      for t in range(3) for p in range(4) for role in (0, 1)
                      for w in (0, 1) for o in (0, 1)]
    raw = np.zeros((2, len(synthetic_rows), 60, 2), np.float32)
    raw[:, :, 0, 0] = [2*r["wording"]-1 for r in synthetic_rows]
    b = np.zeros((len(synthetic_rows), 6), np.float32)
    b[:, 0] = 2
    b[:, 2:4] = 1
    cb = b.copy(); cb[:, 0] = 0
    data = {name: dict(raw=raw.copy(), behavior=b.copy()) for name in ("clean", "wording", "self")}
    data["corrupt"] = dict(raw=raw.copy(), behavior=cb)
    known = {"recovery": 2., "margin": -1., "mean_absolute_axis_difference": 0.,
             "mean_full_signature_distance": 0.}
    oracle, boot = {}, ExplicitBootstrap(synthetic_rows, draws=200)
    for key, (values, mask) in primary_values(synthetic_rows, data, dict(scale=np.ones((2, 60)))).items():
        estimate = boot.interval(values, mask, multiplicity=16)
        assert estimate == dict(mean=known[key[0]], ci95=[known[key[0]]]*2,
                                adjusted_ci=[known[key[0]]]*2, valid=True)
        cursor = oracle
        for part in key[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[key[-1]] = estimate
    synthetic = audit_primary(synthetic_rows, data, dict(scale=np.ones((2, 60))),
                              dict(primary=oracle), draws=200)
    assert synthetic["endpoints"] == 16 and synthetic["seed"] == 2026100409
    # Reject accidental Medium metadata or an obsolete three-column source shape.
    identity = dict(model="gpt2-small", model_spec=SPEC, source=SPEC["source_heads"], layers=SPEC["layers"],
                    batch_size=8, threads=4, head_chunk=8, dtype="float32", kind="natural",
                    bank_sha256=None, delta_sha256=None)
    natural = {key: np.zeros(shape, np.float32) for key, shape in
               dict(behavior=(2, 6), z=(2, 5, 12, 64), head_results=(2, 4, 768), state=(2, 768)).items()}
    audit_small_identity(identity, 2, natural)
    for wrong in (dict(identity, model="gpt2-medium"), dict(identity, source=SPEC["source_heads"][:3])):
        try:
            audit_small_identity(wrong, 2, natural)
        except AssertionError:
            pass
        else:
            raise AssertionError("Wrong checkpoint/source metadata accepted")
    # Frozen old data provide a full external fixture with an independently recorded family.
    old = ROOT / "studies/bridge-005"
    sel = read(old / "data/selection.json")
    rows = selected_rows(read(old / "data/candidates.json"), "test", sel["families"])
    data = {name: arrays(old / "results/test" / (name + ".npz")) for name in ("clean", "corrupt", "wording", "self")}
    checked = audit_primary(rows, data, sel["reference"], read(old / "results/analysis/summary.json"),
                            seed=2026090905, multiplicity=12, original_only=True)
    return dict(status="passed", scope="No new inference: all 12 original Bridge005 point estimates and 50,000-draw adjusted intervals",
                delta_factors_verified=["self", "wording", "order", "role", "order_role", "zero", "four_row_calibration_wording"],
                known_answer_small_endpoints=synthetic["endpoints"], small_signature_coordinates=60,
                max_point_error=checked["max_point_error"], max_adjusted_interval_error=checked["max_adjusted_interval_error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results/verification.json")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test())); return
    status_path = HERE / "results/status.json"
    if not status_path.exists():
        print(json.dumps(dict(status="not_completed", report_written=False))); return
    status = read(status_path)
    if status["status"] not in ("completed", "inference_completed"):
        print(json.dumps(dict(status="not_completed", report_written=False))); return
    if status["test_measured"] and not (HERE / "results/analysis/summary.json").exists():
        print(json.dumps(dict(status="analysis_not_completed", report_written=False))); return
    hasher = Hashes()
    freezes, lookup = audit_freezes(hasher)
    candidates = read(HERE / "data/candidates.json")
    preparation = audit_preparation(candidates, hasher)
    records = audit_runner_records(candidates, hasher, lookup)
    screen = audit_screen(candidates)
    checks = dict(freezes=freezes, runner_records=records, screen_reproduced=True, preparation=preparation,
                  pre_inference_amendment=audit_amendment(hasher))
    calibration = read(HERE / "results/calibration.json")
    assert calibration["passed"] and calibration["tolerance"] == .0005
    assert "secondary-analysis-freeze.json" in freezes
    secondary_time = datetime.fromisoformat(read(HERE / "data/secondary-analysis-freeze.json")["utc"])
    assert secondary_time <= datetime.fromisoformat(read(HERE / "results/calibration/natural-record.json")["started"])
    for name in ("clean", "corrupt", "edited"):
        fixture = arrays(HERE / "results/calibration" / (name + "-validation.npz"))
        assert fixture["original"].shape == fixture["prefix"].shape == (2, 4 if name == "edited" else 8, 60, 2)
        np.testing.assert_allclose(fixture["original"], fixture["prefix"], rtol=0, atol=.0005)
        error = float(np.max(abs(fixture["original"]-fixture["prefix"])))
        recorded = calibration["checks"]["wording" if name == "edited" else name]
        assert error == recorded["signature"] and max(recorded.values()) < .0005
        if "original_behavior" in fixture:
            assert fixture["original_behavior"].shape == fixture["prefix_behavior"].shape == (8, 6)
            np.testing.assert_allclose(fixture["original_behavior"], fixture["prefix_behavior"], rtol=0, atol=.0005)
            assert float(np.max(abs(fixture["original_behavior"]-fixture["prefix_behavior"]))) == recorded["behavior"]
    checks["calibration_arrays_rechecked"] = True
    checks["calibration_scope"] = dict(clean_corrupt_rows=8, edited_rows=4,
        edited_behavior="Recorded maximum checked; no full behavior fixture was saved")
    calibration_source = delta_source(HERE / "results/calibration", {r["id"]: r for r in candidates}, calibration=True)
    assert len(calibration_source["rows"]) == 4
    checks["calibration_wording_delta"] = dict(
        status="reconstructed_only_no_saved_delta_hash", n=4,
        delta_sha256=array_hash(float32_delta(calibration_source["rows"], calibration_source["base"], "wording")),
        source_aggregate_hashes=calibration_source["source_aggregate_hashes"],
        scope="The edited calibration signature was a direct call; no separate delta input record exists")
    outcome = status["outcome"]
    assert outcome in ("no_screen_pair", "no_selection_pair", "independent_qualification_failed", "final_test_completed")
    checks["record_coverage"] = audit_record_coverage(records, outcome)
    expected_freezes = {"input-freeze.json", "secondary-analysis-freeze.json", "screen-freeze.json"}
    if outcome != "no_screen_pair":
        expected_freezes.add("selection-freeze.json")
    if outcome == "final_test_completed":
        expected_freezes.add("test-freeze.json")
    assert set(freezes) == expected_freezes
    test_files = list((HERE / "results/test").glob("*.npz"))
    if outcome == "no_screen_pair":
        assert not screen["pairs"] and not test_files and not (HERE / "data/selection.json").exists()
    else:
        assert screen["pairs"]
        sel, checks["selection"] = audit_selection(candidates, screen)
        if outcome == "no_selection_pair":
            assert sel["status"] == "no_qualified_pair" and not test_files
            assert not (HERE / "results/qualification-summary.json").exists()
        else:
            assert sel["status"] == "selected"
            rows = selected_rows(candidates, "qualification", sel["families"])
            qa = arrays(HERE / "results/qualification/clean.npz")
            q = read(HERE / "results/qualification-summary.json")
            checks["qualification"] = audit_qualification(rows, qa["raw"], qa["behavior"], sel["reference"], q)
            replay = read(HERE / "results/selected-pair-replay.json")
            assert replay["passed"] and max(replay["errors"].values()) < .0005 and min(replay["nonzero"]) > 0
            assert replay["n"] == 32 and replay["families"] == FAMILIES and replay["directions"] == "both"
            if outcome == "independent_qualification_failed":
                assert not q["qualified"] and not test_files and not (HERE / "data/test-freeze.json").exists()
            else:
                assert outcome == "final_test_completed" and status["test_measured"] and q["qualified"]
                rows = selected_rows(candidates, "test", sel["families"])
                data = {name: arrays(HERE / "results/test" / (name + ".npz")) for name in ("clean", "corrupt", "wording", "self")}
                summary = read(HERE / "results/analysis/summary.json")
                assert summary["model"] == "gpt2-small" and summary["signature_coordinates"] == 60
                assert summary["families"] == FAMILIES and summary["n"] == 3200
                assert summary["blocks"] == summary["triples"] == 20 and summary["primary_multiplicity"] == K
                assert summary["qualification"] == q
                for name, expected in summary["input_hashes"].items():
                    assert hasher(ROOT / name) == expected, name
                checks["test_qualification"] = audit_qualification(rows, data["clean"]["raw"], data["clean"]["behavior"],
                                                                 sel["reference"], summary["test_qualification"])
                checks["primary"] = audit_primary(rows, data, sel["reference"], summary)
                checks["final_decisions"] = audit_final_decisions(summary, checks["primary"]["reproduced"])
                for name in ("clean", "corrupt"):
                    natural = arrays(HERE / "results/test" / ("natural-" + name + ".npz"))
                    np.testing.assert_allclose(data[name]["behavior"], natural["behavior"], rtol=0, atol=.0005)
                zero = arrays(HERE / "results/test/zero.npz")["behavior"]
                np.testing.assert_allclose(zero, data["corrupt"]["behavior"], rtol=0, atol=.0005)
                computational = read(HERE / "results/computational-replay.json")
                assert computational["passed"] and max(computational["errors"].values()) < .0005
                for name in ("wording", "corrupt"):
                    fresh = arrays(HERE / "results/verification" / (name + ".npz"))
                    np.testing.assert_allclose(fresh["behavior"], data[name]["behavior"], rtol=0, atol=.0005)
                singleton = arrays(HERE / "results/verification/singleton.npz")
                expected_indices = np.array([i for w in (0, 1) for o in (0, 1)
                                             for i in [j for j, r in enumerate(rows)
                                                       if r["wording"] == w and r["order"] == o][:2]])
                np.testing.assert_array_equal(singleton["indices"], expected_indices)
                np.testing.assert_allclose(singleton["raw"], data["wording"]["raw"][:, singleton["indices"]], rtol=0, atol=.0005)
                np.testing.assert_allclose(singleton["behavior"], data["wording"]["behavior"][singleton["indices"]], rtol=0, atol=.0005)
    assert bool(status["test_measured"]) == (outcome == "final_test_completed")
    result = dict(status="passed", utc=datetime.now(timezone.utc).isoformat(), outcome=outcome,
                  scope="Independent saved-observation computational verification; no new scientific estimands or model inference",
                  helper_sha256=hasher(Path(__file__)), stopping_branch_verified=True,
                  verified_unique_files=len(hasher.cache), checks=checks)
    if outcome == "final_test_completed":
        result["analysis_summary_sha256"] = hasher(HERE / "results/analysis/summary.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status="passed", outcome=outcome, output=str(args.output),
                         verified_unique_files=len(hasher.cache), runner_records=len(records))))


if __name__ == "__main__":
    main()
