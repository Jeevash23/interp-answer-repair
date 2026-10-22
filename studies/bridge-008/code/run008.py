"""Prepared Bridge 008 wider-source runner. No inference occurs on import.

Run only after Bridge007 final analysis confirms the clean gates and a separate
Bridge008 input freeze has been created. The old rows/assay remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
B007=ROOT/"studies/bridge-007"
sys.path[:0]=[str(ROOT/"src"),str(B007/"code")]
from runner007 import make_engine,save_arrays,load_arrays,read,write,digest,now,batches
from analyze007 import rows_for,validate_rows
from common005 import hooks as narrow_hooks
from cpu_engine import Engine003
from operators008 import (LAYERS,layer_sums_from_cache,narrow_layer_deltas,match_broad_to_narrow,
                          make_layer_hooks,matching_summary,self_test)

TOL=.0005
CONDITIONS=("broad_wording","broad_self","matched_wording","matched_self")
SOURCE_FILES=[Path(__file__),HERE/"code/operators008.py",B007/"code/runner007.py",
              ROOT/"studies/bridge-006/code/cpu_engine.py",ROOT/"src/transport_bridge/engine.py"]


def verify_freeze(path):
    value=read(path)
    assert value.get("files"),path
    for name,expected in value["files"].items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        assert digest(p)==(expected["sha256"] if isinstance(expected,dict) else expected),name
    return digest(path)


def prerequisite():
    assert read(B007/"results/inference-complete.json")["status"]=="completed"
    assert read(B007/"results/selected-pair-replay.json")["passed"]
    assert read(B007/"results/verification.json")["status"]=="passed", \
        "Bridge007 independent verification must pass before this extension"
    summary=read(B007/"results/analysis/summary.json")
    assert summary["qualification"]["qualified"] and summary["test_qualification"]["qualified"], \
        "Exploratory extension requires the final clean gates, regardless of primary outcomes"
    selection=read(B007/"data/selection.json")
    rows=rows_for("test",selection["families"])
    assert selection["families"]==summary["families"]
    validate_rows(rows)
    return selection,rows


def checked_b007(name,rows):
    p=B007/"results/test"/(name+".npz")
    record=read(p.with_name(name+"-record.json"))
    assert record["status"]=="completed" and record["ids"]==[r["id"] for r in rows]
    assert digest(p)==record["aggregate_sha256"]
    return load_arrays(p)


def array_hash(value):
    value=np.ascontiguousarray(value)
    h=hashlib.sha256(str(value.dtype).encode()+json.dumps(value.shape).encode())
    h.update(memoryview(value).cast("B"));return h.hexdigest()


def identity(e,rows,freeze,kind,bank=None,layer_delta=None):
    assert rows and len({r["id"] for r in rows})==len(rows)
    if layer_delta is not None:
        assert layer_delta.shape==(len(rows),7,1024) and layer_delta.dtype==np.float32
        assert np.isfinite(layer_delta).all()
    value=dict(schema=1,operator_representation="seven_layer_residual_sums_at_s2",layer_ids=list(LAYERS),
               kind=kind,freeze_sha256=verify_freeze(freeze),
               rows_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
               bank_sha256=None if bank is None else array_hash(bank),
               layer_delta_sha256=None if layer_delta is None else array_hash(layer_delta),
               model="gpt2-medium",threads=torch.get_num_threads(),batch_size=8,
               head_chunk=e.config["head_chunk"],dtype="float32",
               package_versions={x:importlib.metadata.version(x) for x in ["torch","numpy","transformers","transformer-lens"]},
               source_sha256={str(p.relative_to(ROOT)):digest(p) for p in SOURCE_FILES})
    key=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return value,key


def measure(e,rows,path,freeze,kind,bank=None,layer_delta=None):
    """Hash-checked, atomic chunks. A layer axis is never treated as a head axis."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rp=path.with_name(path.stem+"-record.json")
    meta,key=identity(e,rows,freeze,kind,bank,layer_delta)
    if rp.exists():
        state=read(rp);assert state["inputs"]==meta and state["input_key"]==key,"Resume inputs changed"
    else:
        assert not path.exists(),path
        state=dict(started=now(),status="running",inputs=meta,input_key=key,ids=[r["id"] for r in rows],hashes={})
        write(rp,state)
    if state["status"]=="completed":
        assert digest(path)==state["aggregate_sha256"]
        return load_arrays(path)
    output={}
    for part,ii in enumerate(batches(rows,8)):
        chunk=path.with_name(f"{path.stem}-chunk{part:04d}.npz")
        if chunk.exists():
            if chunk.name in state["hashes"]:assert digest(chunk)==state["hashes"][chunk.name]
            with np.load(chunk,allow_pickle=False) as z:
                assert str(z["_resume_input_key"].item())==key
                np.testing.assert_array_equal(z["_resume_indices"],ii)
                result={k:z[k] for k in z.files if not k.startswith("_resume_")}
        else:
            assert chunk.name not in state["hashes"],chunk
            started=time.perf_counter()
            rr=[dict(rows[i],token_ids=rows[i]["corrupt_tokens"]) if layer_delta is not None else rows[i] for i in ii]
            if kind=="natural":
                behavior,cache=e.forward(rr)
                result=dict(layer_sums=layer_sums_from_cache(e,rr,cache),behavior=behavior.cpu().numpy(),
                            layer_ids_per_row=np.tile(np.asarray(LAYERS,np.int64),(len(ii),1)))
            else:
                assert kind=="signature" and bank is not None and layer_delta is not None
                hs=make_layer_hooks(e,layer_delta[ii],rr)
                raw,behavior=e.signatures(rr,torch.as_tensor(bank),hs)
                result=dict(raw=raw,behavior=behavior)
            assert all(np.isfinite(v).all() for v in result.values())
            save_arrays(chunk,**result,_resume_input_key=np.array(key),_resume_indices=ii)
            state["last_chunk_seconds"]=time.perf_counter()-started
        for name,value in result.items():
            axis=1 if name=="raw" else 0
            assert value.shape[axis]==len(ii)
            if name not in output:
                shape=list(value.shape);shape[axis]=len(rows);output[name]=np.empty(shape,value.dtype)
            if axis:output[name][:,ii]=value
            else:output[name][ii]=value
        state["hashes"][chunk.name]=digest(chunk);state["completed_chunks"]=part+1;write(rp,state)
        if part%8==0:print(now(),path.stem,"chunk",part+1,"rows",len(rows),flush=True)
    if path.exists():
        old=load_arrays(path);assert old.keys()==output.keys()
        for key in old:np.testing.assert_array_equal(old[key],output[key])
    else:save_arrays(path,**output)
    state.update(status="completed",finished=now(),aggregate_sha256=digest(path));write(rp,state)
    return output


def _save_fixed(path,arrays):
    if path.exists():
        old=load_arrays(path);assert old.keys()==arrays.keys()
        for key in old:np.testing.assert_array_equal(old[key],arrays[key])
    else:save_arrays(path,**arrays)


def prepare(e,selection,rows,freeze):
    folder=HERE/"results/natural"
    clean=measure(e,rows,folder/"clean.npz",freeze,"natural")
    cr=[dict(r,token_ids=r["corrupt_tokens"]) for r in rows]
    corrupt=measure(e,cr,folder/"corrupt.npz",freeze,"natural")
    old_clean,old_corrupt=[checked_b007("natural-"+k,rows) for k in ("clean","corrupt")]
    for current,previous in [(clean,old_clean),(corrupt,old_corrupt)]:
        np.testing.assert_allclose(current["behavior"],previous["behavior"],rtol=0,atol=TOL)
        np.testing.assert_array_equal(current["layer_ids_per_row"],np.tile(LAYERS,(len(rows),1)))
    narrow=narrow_layer_deltas(old_clean["head_results"],old_corrupt["head_results"],e.source)
    broad=clean["layer_sums"]-corrupt["layer_sums"]
    donors=validate_rows(rows);summary={}
    for condition,indices in [("wording",donors),("self",np.arange(len(rows)))]:
        arrays=match_broad_to_narrow(broad[indices],narrow[indices])
        out=HERE/"results/deltas"/(condition+".npz");_save_fixed(out,arrays)
        summary[condition]=matching_summary(arrays,rows)
        summary[condition]["delta_sha256"]=digest(out)
    target=HERE/"results/deltas/matching-summary.json"
    value=dict(utc=now(),matching=summary,freeze_sha256=digest(freeze),
               narrow_semantics="Actual float32 B007 head differences and within-layer torch sums",
               broad_semantics="Float32 clean-minus-corrupt sums of all sixteen projected heads per layer")
    if target.exists():
        old=read(target);assert old["matching"]==summary and old["freeze_sha256"]==digest(freeze)
    else:write(target,value)


def calibration(e,selection,freeze):
    """Small real checks scheduled after B007; never run during code preparation."""
    target=HERE/"results/calibration.json"
    if target.exists():
        value=read(target);assert value["passed"] and value["freeze_sha256"]==digest(freeze);return
    rows=rows_for("calibration",selection["families"])
    rows=[next(r for r in rows if r["wording"]==w and r["order"]==o and r["role"]==role)
          for w in range(2) for o in range(2) for role in range(2)]
    bank=torch.tensor(load_arrays(B007/"results/fit/bank.npz")["bank"])
    collected={};checks={}
    for condition in ("clean","corrupt"):
        rr=rows if condition=="clean" else [dict(r,token_ids=r["corrupt_tokens"]) for r in rows]
        behavior,cache=e.forward(rr);_,_,heads=e.extract(rr,cache)
        sums=layer_sums_from_cache(e,rr,cache)
        pos=torch.tensor([r["s2"] for r in rr]);ix=torch.arange(len(rr))
        full=torch.stack([torch.stack([cache[f"blocks.{l}.attn.hook_z"][ix,pos,h] @ e.weights[l,h]
                                     for h in range(16)],1) for l in LAYERS],1)
        error=float(np.max(abs(full.sum(2).numpy()-sums)))
        checks[condition+"_explicit_head_sum_error"]=error;assert error<TOL
        collected[condition]=dict(sums=sums,heads=heads.numpy(),full=full.numpy())
    broad=collected["clean"]["sums"]-collected["corrupt"]["sums"]
    explicit=torch.from_numpy(collected["clean"]["full"]-collected["corrupt"]["full"]).sum(2).numpy()
    checks["explicit_112_head_delta_error"]=float(np.max(abs(broad-explicit)))
    assert checks["explicit_112_head_delta_error"]<TOL
    narrow=narrow_layer_deltas(collected["clean"]["heads"],collected["corrupt"]["heads"],e.source)
    donors=validate_rows(rows);cr=[dict(r,token_ids=r["corrupt_tokens"]) for r in rows]
    # Narrow layer-sum hooks must reproduce the inherited twelve-head hook operator.
    hd=torch.tensor(collected["clean"]["heads"]-collected["corrupt"]["heads"])[donors]
    a=e.forward(cr,narrow_hooks(e,hd,cr))[0]
    b=e.forward(cr,make_layer_hooks(e,narrow[donors],cr))[0]
    checks["narrow_layer_operator_behavior_error"]=float((a-b).abs().max());assert checks["narrow_layer_operator_behavior_error"]<TOL
    _save_fixed(HERE/"results/calibration/source-validation.npz",dict(
        clean_layer_sums=collected["clean"]["sums"],corrupt_layer_sums=collected["corrupt"]["sums"],
        clean_projected_heads=collected["clean"]["full"],corrupt_projected_heads=collected["corrupt"]["full"],
        narrow_clean_heads=collected["clean"]["heads"],narrow_corrupt_heads=collected["corrupt"]["heads"],
        narrow_layer_deltas=narrow,original_narrow_behavior=a.cpu().numpy(),layer_narrow_behavior=b.cpu().numpy(),
        donor_indices=donors,layer_ids=np.asarray(LAYERS),selected_source_heads=np.asarray(e.source),
        ids=np.asarray([r["id"] for r in rows])))
    matched=match_broad_to_narrow(broad[donors],narrow[donors]);assert matched["matched_valid"].all()
    # Both words/directions/orders/roles occur across this fixed eight-row check.
    for name,d in [("broad_wording",broad[donors]),("matched_wording",matched["layer_deltas"])]:
        hs=make_layer_hooks(e,d,cr)
        original,ob=Engine003.signatures(e,cr,bank,hs)
        prefix,pb=e.signatures(cr,bank,hs)
        checks[name+"_signature_error"]=float(np.max(abs(original-prefix)))
        checks[name+"_behavior_error"]=float(np.max(abs(ob-pb)))
        assert max(checks[name+"_signature_error"],checks[name+"_behavior_error"])<TOL
        _save_fixed(HERE/"results/calibration"/(name+".npz"),dict(original=original,prefix=prefix,original_behavior=ob,prefix_behavior=pb))
        if name=="broad_wording":
            # Vector agreement alone is insufficient: explicitly summed head
            # differences must also reproduce the efficient operator's output.
            explicit_raw,explicit_behavior=e.signatures(cr,bank,make_layer_hooks(e,explicit[donors],cr))
            checks["explicit_112_operator_signature_error"]=float(np.max(abs(explicit_raw-prefix)))
            checks["explicit_112_operator_behavior_error"]=float(np.max(abs(explicit_behavior-pb)))
            assert max(checks["explicit_112_operator_signature_error"],checks["explicit_112_operator_behavior_error"])<TOL
            _save_fixed(HERE/"results/calibration/explicit112-prefix.npz",dict(
                explicit_raw=explicit_raw,efficient_raw=prefix,
                explicit_behavior=explicit_behavior,efficient_behavior=pb,
                explicit_layer_delta=explicit[donors],efficient_layer_delta=broad[donors]))
    write(target,dict(utc=now(),passed=True,n=len(rows),ids=[r["id"] for r in rows],checks=checks,
                      tolerance=TOL,freeze_sha256=digest(freeze)))


def inference(e,selection,rows,freeze):
    assert read(HERE/"results/calibration.json")["passed"]
    bank=load_arrays(B007/"results/fit/bank.npz")["bank"]
    matching=read(HERE/"results/deltas/matching-summary.json")
    assert matching["freeze_sha256"]==digest(freeze)
    outcomes={}
    for condition in CONDITIONS:
        kind,donor=condition.split("_")
        path=HERE/"results/deltas"/(donor+".npz")
        assert digest(path)==matching["matching"][donor]["delta_sha256"]
        arrays=load_arrays(path);np.testing.assert_array_equal(arrays["layer_ids"],LAYERS)
        if kind=="matched" and not arrays["matched_valid"].all():
            value=dict(condition=condition,status="non_evaluable",matching=matching["matching"][donor],
                       freeze_sha256=digest(freeze),reason="At least one undefined norm match; no rows dropped")
            marker=HERE/"results/test"/(condition+"-non-evaluable.json")
            if marker.exists():assert read(marker)==value
            else:write(marker,value)
            outcomes[condition]="non_evaluable";continue
        layer_delta=arrays["layer_deltas" if kind=="matched" else "broad_layer_deltas"]
        measure(e,rows,HERE/"results/test"/(condition+".npz"),freeze,"signature",bank,layer_delta)
        outcomes[condition]="completed"
    write(HERE/"results/inference-complete.json",dict(utc=now(),status="completed",conditions=outcomes,
                                                     freeze_sha256=digest(freeze),n=len(rows)))


def computational_replay(selection,rows,freeze):
    """Fresh-engine singleton-head replay on eight manifest-selected rows only."""
    target=HERE/"results/computational-replay.json"
    if target.exists():
        value=read(target);assert value["passed"] and value["freeze_sha256"]==digest(freeze);return
    completed=read(HERE/"results/inference-complete.json")
    assert completed["status"]=="completed" and completed["freeze_sha256"]==digest(freeze)
    indices=np.asarray([i for w in range(2) for o in range(2)
                        for i in [j for j,r in enumerate(rows) if r["wording"]==w and r["order"]==o][:2]])
    assert len(indices)==8
    chosen=[dict(rows[i],token_ids=rows[i]["corrupt_tokens"]) for i in indices]
    bank=torch.tensor(load_arrays(B007/"results/fit/bank.npz")["bank"])
    e=make_engine(4);e.config["head_chunk"]=1
    checks={};errors={}
    for condition in CONDITIONS:
        if completed["conditions"][condition]=="non_evaluable":
            marker=HERE/"results/test"/(condition+"-non-evaluable.json")
            assert read(marker)["status"]=="non_evaluable"
            checks[condition]=dict(status="non_evaluable",marker_sha256=digest(marker));continue
        kind,donor=condition.split("_")
        arrays=load_arrays(HERE/"results/deltas"/(donor+".npz"))
        delta=arrays["layer_deltas" if kind=="matched" else "broad_layer_deltas"][indices]
        path=HERE/"results/test"/(condition+".npz")
        record=read(path.with_name(condition+"-record.json"))
        assert record["status"]=="completed" and digest(path)==record["aggregate_sha256"]
        expected=load_arrays(path)
        raw,behavior=e.signatures(chosen,bank,make_layer_hooks(e,delta,chosen))
        signature_error=float(np.max(abs(raw-expected["raw"][:,indices])))
        behavior_error=float(np.max(abs(behavior-expected["behavior"][indices])))
        assert max(signature_error,behavior_error)<TOL,(condition,signature_error,behavior_error)
        saved=HERE/"results/verification"/(condition+"-singleton.npz")
        _save_fixed(saved,dict(indices=indices,raw=raw,behavior=behavior,layer_deltas=delta))
        checks[condition]=dict(status="passed",signature_error=signature_error,
                               behavior_error=behavior_error,saved_sha256=digest(saved))
        errors[condition+"_signature"]=signature_error;errors[condition+"_behavior"]=behavior_error
    write(target,dict(utc=now(),passed=True,fresh_engine=True,head_chunk=1,n=len(indices),
                      ids=[rows[i]["id"] for i in indices],indices=indices.tolist(),
                      checks=checks,errors=errors,tolerance=TOL,freeze_sha256=digest(freeze)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["self-test","calibration","prepare","inference","computational_replay","pipeline"])
    parser.add_argument("--freeze",type=Path,default=HERE/"data/input-freeze.json")
    parser.add_argument("--output",type=Path,help="Optional synthetic self-test JSON report")
    args=parser.parse_args()
    if args.stage=="self-test":
        result=self_test()
        if args.output:write(args.output,dict(result,utc=now(),source_hashes={str(p.relative_to(ROOT)):digest(p) for p in SOURCE_FILES}))
        print(json.dumps(result));return
    assert args.output is None,"--output only applies to synthetic self-test"
    verify_freeze(args.freeze);selection,rows=prerequisite()
    with torch.inference_mode():
        if args.stage=="computational_replay":
            computational_replay(selection,rows,args.freeze);return
        e=make_engine(4)
        if args.stage in ("calibration","pipeline"):calibration(e,selection,args.freeze)
        if args.stage in ("prepare","pipeline"):prepare(e,selection,rows,args.freeze)
        if args.stage in ("inference","pipeline"):inference(e,selection,rows,args.freeze)
        if args.stage=="pipeline":
            del e
            computational_replay(selection,rows,args.freeze)


if __name__=="__main__":main()
