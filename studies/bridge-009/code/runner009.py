"""Exact, resumable numerical primitives for the prospectively frozen Bridge 009.

Scientific selection, reference fitting and decision rules live in the pipeline.
This module only measures supplied rows with the validated Bridge 006 prefix
algorithm, retaining both replacement references and both original score types.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "studies/bridge-006/code")]
from small_engine import SmallCPUEngine
from common005 import batches, hooks, now, read, write
from activation_transfer.paths import verify_frozen_file

BATCH_SIZE = 8
SOURCE_FILES = [
    Path(__file__), HERE / "code/small_engine.py", ROOT / "studies/bridge-006/code/cpu_engine.py",
    ROOT / "studies/bridge-005/code/common005.py",
    ROOT / "studies/bridge-003/code/common003.py",
    ROOT / "studies/bridge-002/code/assay.py",
    ROOT / "src/transport_bridge/engine.py", ROOT / "configs/study.json",
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files if not k.startswith("_resume_")}


def save_arrays(path, **arrays):
    """Replace atomically, so a terminated process cannot publish half an array."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(value.shape).encode())
    h.update(memoryview(value).cast("B"))
    return h.hexdigest()


def verify_freeze(path):
    path = Path(path).resolve()
    frozen = read(path)
    assert isinstance(frozen.get("files"), dict) and frozen["files"], path
    for name, expected in frozen["files"].items():
        target = Path(name)
        if not target.is_absolute():
            candidates = [ROOT / name, path.parent / name, HERE / name]
            target = next((p for p in candidates if p.is_file()), candidates[0])
        if isinstance(expected, dict):
            expected = expected["sha256"]
        verify_frozen_file(target, expected)
    return digest(path)


def make_engine(threads=4):
    engine = SmallCPUEngine()
    torch.set_num_threads(threads)
    engine.config = dict(engine.config, threads=threads)
    engine.runner_batch_size = BATCH_SIZE
    return engine


def _inputs(engine, rows, freeze_path, kind, bank=None, delta=None):
    assert rows and len({r["id"] for r in rows}) == len(rows)
    assert all(len(r["token_ids"]) == len(r["corrupt_tokens"]) for r in rows)
    if delta is not None:
        assert tuple(delta.shape) == (len(rows), len(engine.source), engine.spec["width"])
        assert np.isfinite(np.asarray(delta)).all()
    if bank is not None:
        assert tuple(bank.shape) == (len(engine.layers), engine.heads, engine.weights.shape[2])
        assert np.isfinite(np.asarray(bank)).all()
    identity = dict(
        schema=1, kind=kind, freeze_sha256=verify_freeze(freeze_path),
        rows_sha256=_json_hash(rows), n=len(rows),
        bank_sha256=None if bank is None else _array_hash(bank),
        delta_sha256=None if delta is None else _array_hash(delta),
        batch_size=engine.runner_batch_size, threads=torch.get_num_threads(),
        head_chunk=engine.config["head_chunk"], dtype="float32",
        package_versions={name: importlib.metadata.version(name) for name in
                          ["numpy", "torch", "transformers", "transformer-lens"]},
        model="gpt2-small", source=engine.source, layers=engine.layers,
        model_spec=engine.spec, configuration=engine.config,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in SOURCE_FILES},
    )
    # JSON-normalise tuples so an unchanged resume compares identically.
    return json.loads(json.dumps(identity)), _json_hash(identity)


def _measure(engine, rows, output_path, freeze_path, kind, fn, bank=None, delta=None):
    target = Path(output_path)
    assert target.suffix == ".npz", target
    target.parent.mkdir(parents=True, exist_ok=True)
    record_path = target.with_name(target.stem + "-record.json")
    identity, key = _inputs(engine, rows, freeze_path, kind, bank, delta)
    if record_path.exists():
        state = read(record_path)
        assert state["input_key"] == key and state["inputs"] == identity, "Resume inputs changed"
    else:
        assert not target.exists(), "Aggregate exists without its provenance record"
        state = dict(started=now(), input_key=key, inputs=identity,
                     ids=[r["id"] for r in rows], hashes={}, status="running")
        write(record_path, state)
    if state["status"] == "completed":
        assert digest(target) == state["aggregate_sha256"], target
        return load_arrays(target)
    result = {}
    for part, ids in enumerate(batches(rows, engine.runner_batch_size)):
        chunk = target.with_name(f"{target.stem}-chunk{part:04d}.npz")
        if chunk.exists():
            # A chunk can be atomically saved just before a crash prevents the
            # record update. Its embedded input key/indices allow safe recovery.
            if chunk.name in state["hashes"]:
                assert digest(chunk) == state["hashes"][chunk.name], chunk
            with np.load(chunk, allow_pickle=False) as cached:
                assert str(cached["_resume_input_key"].item()) == key, chunk
                np.testing.assert_array_equal(cached["_resume_indices"], ids)
                values = {k: cached[k] for k in cached.files if not k.startswith("_resume_")}
        else:
            assert chunk.name not in state["hashes"], "Recorded chunk missing"
            started = time.perf_counter()
            values = fn(ids)
            assert values and all(np.isfinite(v).all() for v in values.values()), chunk
            save_arrays(chunk, **values, _resume_input_key=np.array(key), _resume_indices=ids)
            state["last_chunk_seconds"] = time.perf_counter() - started
        for name, value in values.items():
            axis = 1 if name == "raw" else 0
            assert value.shape[axis] == len(ids), (name, value.shape, ids)
            if name not in result:
                shape = list(value.shape)
                shape[axis] = len(rows)
                result[name] = np.empty(shape, dtype=value.dtype)
            if axis == 1:
                result[name][:, ids] = value
            else:
                result[name][ids] = value
        state["hashes"][chunk.name] = digest(chunk)
        state["completed_chunks"] = part + 1
        write(record_path, state)
        if part % 8 == 0:
            print(now(), target.stem, "chunk", part + 1, "rows", len(rows), flush=True)
    if target.exists():
        previous = load_arrays(target)
        assert previous.keys() == result.keys()
        for name in result:
            np.testing.assert_array_equal(previous[name], result[name])
    else:
        save_arrays(target, **result)
    state.update(status="completed", finished=now(), aggregate_sha256=digest(target))
    write(record_path, state)
    return result


@torch.inference_mode()
def measure_natural(e, rows, output_path, freeze_path):
    """Measure token_ids as supplied; caller can supply clean or corrupted rows."""
    def run(ids):
        rr = [rows[i] for i in ids]
        behavior, cache = e.forward(rr)
        state, z, heads = e.extract(rr, cache)
        return dict(behavior=behavior.cpu().numpy(), z=z.numpy(),
                    head_results=heads.numpy(), state=state.numpy())
    return _measure(e, rows, output_path, freeze_path, "natural", run)


@torch.inference_mode()
def measure_signatures(e, rows, bank, output_path, freeze_path, d=None):
    bank = torch.as_tensor(bank, dtype=torch.float32, device="cpu")
    d = None if d is None else torch.as_tensor(d, dtype=torch.float32, device="cpu")
    def run(ids):
        rr = [dict(rows[i], token_ids=rows[i]["corrupt_tokens"]) if d is not None
              else rows[i] for i in ids]
        hs = () if d is None else hooks(e, d[ids], rr)
        raw, behavior = e.signatures(rr, bank, hs)
        return dict(raw=raw, behavior=behavior)
    return _measure(e, rows, output_path, freeze_path, "signature", run, bank, d)


@torch.inference_mode()
def measure_behavior(e, rows, output_path, freeze_path, d=None):
    d = None if d is None else torch.as_tensor(d, dtype=torch.float32, device="cpu")
    def run(ids):
        rr = [dict(rows[i], token_ids=rows[i]["corrupt_tokens"]) if d is not None
              else rows[i] for i in ids]
        hs = () if d is None else hooks(e, d[ids], rr)
        return dict(behavior=e.forward(rr, hs)[0].cpu().numpy())
    return _measure(e, rows, output_path, freeze_path, "behavior", run, delta=d)["behavior"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["natural", "signatures", "behavior"])
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--bank", type=Path, help="NPZ containing bank, or z to average")
    parser.add_argument("--delta", type=Path, help="NPZ containing a prepared delta array")
    parser.add_argument("--corrupt", action="store_true", help="Use corrupt tokens when no delta supplied")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    assert args.threads > 0 and args.batch_size > 0
    rows = read(args.rows)
    if args.corrupt:
        assert args.delta is None, "A supplied edit already operates on corrupted rows"
        rows = [dict(r, token_ids=r["corrupt_tokens"]) for r in rows]
    verify_freeze(args.freeze)
    engine = make_engine(args.threads)
    engine.runner_batch_size = args.batch_size
    d = None if args.delta is None else load_arrays(args.delta)["delta"]
    if args.stage == "natural":
        assert d is None
        measure_natural(engine, rows, args.out, args.freeze)
    elif args.stage == "behavior":
        measure_behavior(engine, rows, args.out, args.freeze, d)
    else:
        assert args.bank is not None
        arrays = load_arrays(args.bank)
        bank = arrays["bank"] if "bank" in arrays else arrays["z"].mean(0)
        measure_signatures(engine, rows, bank, args.out, args.freeze, d)


if __name__ == "__main__":
    main()
