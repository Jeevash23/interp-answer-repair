"""Prespecified secondary Bridge 007 analyses; saved observations only.

All intervals are pointwise 95% crossed-bootstrap intervals. No secondary result
replaces a primary decision. This implementation is separately frozen before
final transfer, without changing the main study's frozen sources.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze007 import (Crossed, DRAWS, SEED, REFS, cell_masks, fit_reference,
                        paired, rows_for, validate_rows)

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
FREEZE = HERE / "data/secondary-analysis-freeze.json"
CONDITIONS = ["clean", "corrupt", "wording", "self", "zero", "order", "role", "order_role"]
CONDITIONS += [kind + str(i) for kind in ("residual", "projected") for i in range(3)]
PLAN = {
    "uncertainty": "Pointwise 95% crossed bootstrap, 50000 draws, seed 2026100407; no multiplicity claims",
    "behavior": "Every condition: logit/probability recovery, top1 and pairwise rates; pooled and each wording/order cell",
    "paired_controls": "Wording minus each control/self and each three-draw residual/projected mean: logit and top1 contrasts",
    "recovery_fraction": "Wording and self: ratio of mean recovery to mean clean-minus-corrupt logit gap; paired bootstrap ratios",
    "alternative_distances": "Wording and self: unscaled paired L2, uncentred standardized cosine, centroid-normalized scaled L2 margins",
    "movement": "Wording/self: axial change, displacement, recipient/donor distance reduction, projection toward recipient/donor, cosine with full repair, recipient distance closed, perpendicular squared movement; ideal-repair comparison",
    "injected_delta": "Flattened actual twelve-head wording/self deltas: norms, relative difference and cosine",
    "scope": "All means and intervals pooled and per wording/order cell; unresolved values retained and affected summaries invalidated",
    "files": ["results/secondary-analysis/summary.json", "results/secondary-analysis/per-example.npz"],
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def freeze():
    assert not (HERE / "results/test").exists(), "Secondary implementation must freeze before final inference"
    files = [Path(__file__), HERE / "code/analyze007.py", HERE / "docs/PROTOCOL.md",
             HERE / "data/input-freeze.json"]
    plan_sha = hashlib.sha256(json.dumps(PLAN, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    value = dict(utc=datetime.now(timezone.utc).isoformat(),
                 scope="Secondary implementation freeze during clean-only stages; no Bridge007 scientific arrays inspected",
                 files={str(p.relative_to(ROOT)): digest(p) for p in files},
                 planned_outputs=PLAN, planned_outputs_sha256=plan_sha)
    if FREEZE.exists():
        verify_freeze()
    else:
        write(FREEZE, value)
    return read(FREEZE)


def verify_freeze():
    value = read(FREEZE)
    for name, expected in value["files"].items():
        assert digest(ROOT / name) == expected, name
    expected = hashlib.sha256(json.dumps(PLAN, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert value["planned_outputs_sha256"] == expected


def divide(num, den, tolerance=1e-12):
    return np.asarray(num, float) / np.where(np.abs(den) >= tolerance, den, np.nan)


def pointwise(boot, values, mask=None):
    result = boot.interval(values, mask, multiplicity=1)
    result.pop("adjusted_ci", None)
    result["interval_scope"] = "pointwise 95%"
    return result


def summarize(boot, values, masks):
    return dict(pooled=pointwise(boot, values),
                cells={cell: pointwise(boot, values, mask) for cell, mask in masks.items()})


def ratio_interval(boot, numerator, denominator, mask=None):
    ng, dg = boot.grid(numerator, mask), boot.grid(denominator, mask)
    bn, bd = boot.resample(ng), boot.resample(dg)
    result = dict(n=len(boot.rows) if mask is None else int(mask.sum()),
                  blocks=len(boot.blocks), triples=len(boot.triples), draws=boot.draws,
                  seed=boot.seed, interval_scope="pointwise 95%; ratio of paired bootstrap means",
                  numerator_mean=float(ng.mean()), denominator_mean=float(dg.mean()))
    if (not np.isfinite(bn).all() or not np.isfinite(bd).all()
            or abs(dg.mean()) < 1e-12 or np.any(abs(bd) < 1e-12)
            or bd.min() <= 0 <= bd.max()):
        return dict(result, valid=False, mean=None, ci95=None,
                    reason="Nonfinite or unstable ratio denominator; no bootstrap draws omitted")
    return dict(result, valid=True, mean=float(ng.mean() / dg.mean()),
                ci95=np.quantile(bn / bd, [.025, .975]).tolist())


def alternative_margins(e, r, d, scale, centers, rows):
    unscaled = paired(e, r, d, np.ones_like(scale))["margin"]
    es, rs, ds = e / scale, r / scale, d / scale
    def cosine(a, b):
        return divide((a * b).sum(-1), np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    cosine_margin = cosine(es, ds) - cosine(es, rs)
    order = np.array([row["order"] for row in rows])
    # Fitted centers are already standardized. Direction is irrelevant to norm.
    centroid_norm = np.linalg.norm(centers[:, 2 + order] - centers[:, order], axis=-1)
    centroid_margin = divide(np.linalg.norm(es-rs, axis=-1) - np.linalg.norm(es-ds, axis=-1),
                             centroid_norm, tolerance=1e-6)
    return dict(unscaled_paired_l2=unscaled, standardized_cosine=cosine_margin,
                centroid_normalized_l2=centroid_margin)


def movement_metrics(e, c, r, d, scale):
    v, g, h, gd = (d-r)/scale, (r-c)/scale, (e-c)/scale, (d-c)/scale
    vn, gn, hn, gdn = [np.linalg.norm(x, axis=-1) for x in (v,g,h,gd)]
    along = divide((h*v).sum(-1), vn**2)
    ideal_along = divide((g*v).sum(-1), vn**2)
    perpendicular = h - along[..., None]*v
    ideal_perpendicular = g - ideal_along[..., None]*v
    return dict(
        axial_change=along,
        displacement_clean_units=divide(hn, vn, tolerance=1e-6),
        recipient_distance_reduction_clean_units=divide(gn-np.linalg.norm((e-r)/scale,axis=-1),vn,tolerance=1e-6),
        donor_distance_reduction_clean_units=divide(gdn-np.linalg.norm((e-d)/scale,axis=-1),vn,tolerance=1e-6),
        progress_toward_recipient=divide((h*g).sum(-1),gn**2),
        progress_toward_donor=divide((h*gd).sum(-1),gdn**2),
        cosine_with_ideal_repair=divide((h*g).sum(-1),hn*gn),
        recipient_distance_closed=1-divide(np.linalg.norm((e-r)/scale,axis=-1),gn),
        perpendicular_fraction=divide((perpendicular**2).sum(-1),hn**2),
        ideal_repair_perpendicular_fraction=divide((ideal_perpendicular**2).sum(-1),gn**2),
        ideal_repair_displacement_clean_units=divide(gn,vn,tolerance=1e-6))


def analyze_secondary(selection, rows, data, natural, draws=DRAWS):
    donors = validate_rows(rows)
    boot, masks = Crossed(rows, draws=draws), cell_masks(rows)
    for name in CONDITIONS:
        assert data[name]["behavior"].shape == (len(rows), 6)
        assert np.isfinite(data[name]["behavior"]).all()
    for name in ("clean", "corrupt", "wording", "self"):
        assert data[name]["raw"].shape[:2] == (2, len(rows))
        assert np.isfinite(data[name]["raw"]).all()
    result = dict(scope="Prespecified secondary analyses; no primary decision changes",
                  interval_scope="All intervals pointwise 95% crossed bootstrap",
                  n=len(rows), families=selection["families"], blocks=len(boot.blocks),
                  triples=len(boot.triples), draws=draws, seed=SEED,
                  behavior={}, paired_behavior_contrasts={}, recovery_fraction={},
                  alternative_distances={}, movement={}, injected_delta={})
    metrics = {}
    behavior = {k: data[k]["behavior"].astype(float) for k in CONDITIONS}
    clean_gap = behavior["clean"][:,0]-behavior["corrupt"][:,0]
    result["clean_logit_gap"] = summarize(boot, clean_gap, masks)
    metrics["clean_logit_gap"] = clean_gap
    for name in CONDITIONS:
        values = dict(recovery=behavior[name][:,0]-behavior["corrupt"][:,0],
                      probability_recovery=behavior[name][:,1]-behavior["corrupt"][:,1],
                      top1=behavior[name][:,3], pairwise=behavior[name][:,2])
        result["behavior"][name] = {key: summarize(boot, value, masks) for key,value in values.items()}
        metrics.update({f"behavior__{name}__{key}":value for key,value in values.items()})
    for name in [x for x in CONDITIONS if x not in ("clean","corrupt","wording")]+["residual_mean","projected_mean"]:
        comparator = (np.mean([behavior[name[:-5]+str(i)] for i in range(3)], axis=0)
                      if name.endswith("_mean") else behavior[name])
        values = dict(logit=behavior["wording"][:,0]-comparator[:,0],
                      top1=behavior["wording"][:,3]-comparator[:,3])
        result["paired_behavior_contrasts"]["wording_minus_"+name] = {
            key:summarize(boot,value,masks) for key,value in values.items()}
        metrics.update({f"contrast__wording_minus_{name}__{key}":value for key,value in values.items()})
    for name in ("wording","self"):
        gain = behavior[name][:,0]-behavior["corrupt"][:,0]
        result["recovery_fraction"][name] = dict(pooled=ratio_interval(boot,gain,clean_gap),
            cells={cell:ratio_interval(boot,gain,clean_gap,mask) for cell,mask in masks.items()})
    r = data["clean"]["raw"][...,0].astype(float)
    c = data["corrupt"]["raw"][...,0].astype(float)
    d = r[:,donors]
    scale = np.asarray(selection["reference"]["scale"],float)[:,None]
    centers = np.asarray(selection["reference"]["centers"],float)
    assert np.isfinite(scale).all() and (scale>0).all()
    for name in ("wording","self"):
        e = data[name]["raw"][...,0].astype(float)
        alternatives = alternative_margins(e,r,d,scale,centers,rows)
        result["alternative_distances"][name] = {
            key:{ref:summarize(boot,value[i],masks) for i,ref in enumerate(REFS)}
            for key,value in alternatives.items()}
        movement = movement_metrics(e,c,r,d,scale)
        result["movement"][name] = {
            key:{ref:summarize(boot,value[i],masks) for i,ref in enumerate(REFS)}
            for key,value in movement.items()}
        metrics.update({f"alternative__{name}__{key}":value for key,value in alternatives.items()})
        metrics.update({f"movement__{name}__{key}":value for key,value in movement.items()})
    # Reproduce the actual float32 subtraction used to construct the injected
    # differences, then evaluate their geometry in float64.
    hself = (natural["clean"]["head_results"]-natural["corrupt"]["head_results"]).astype(float)
    assert hself.shape[0] == len(rows) and np.isfinite(hself).all()
    ds = hself.reshape(len(rows),-1); dw = ds[donors]
    ns,nw = np.linalg.norm(ds,axis=-1),np.linalg.norm(dw,axis=-1)
    injected = dict(self_norm=ns,wording_norm=nw,
                    relative_difference=divide(np.linalg.norm(dw-ds,axis=-1),ns),
                    cosine=divide((dw*ds).sum(-1),ns*nw))
    result["injected_delta"] = {key:summarize(boot,value,masks) for key,value in injected.items()}
    metrics.update({"injected__"+key:value for key,value in injected.items()})
    checks = dict(zero_edit_max=float(np.max(abs(behavior["zero"]-behavior["corrupt"]))))
    for name in ("clean","corrupt"):
        checks["natural_"+name+"_behavior_max"] = float(np.max(abs(behavior[name]-natural[name]["behavior"])))
    assert max(checks.values()) < .0005, checks
    result["numerical_checks"] = checks
    result["control_interpretation"] = "Order/role transformations share exact algebraic symmetries; not independent corroborations"
    return result,metrics


def self_test():
    rows=[dict(id=f"{t}-{p}-{role}-{w}-{o}",template=t,pair=p,role=role,wording=w,order=o,family=10+w)
          for t in range(2) for p in range(3) for role in range(2) for w in range(2) for o in range(2)]
    n=len(rows);donors=validate_rows(rows)
    raw=np.zeros((2,n,2,2));raw[:,:,0,0]=1+2*np.array([x["wording"] for x in rows]);raw[:,:,1,0]=1
    r=raw[...,0];v=r[:,donors]-r
    behavior=np.zeros((n,6));behavior[:,0]=2;behavior[:,2:4]=1
    data={"clean":dict(raw=raw.copy(),behavior=behavior.copy())}
    for name,axis,off,gap in [("corrupt",.5,1.,0.),("wording",.25,.5,1.),("self",.225,.45,1.1)]:
        z=raw.copy();z[...,0]+=axis*v;z[:,:,1,0]+=off
        b=behavior.copy();b[:,0]=gap
        data[name]=dict(raw=z,behavior=b)
    for name in CONDITIONS:
        if name not in data:data[name]=dict(behavior=data["corrupt"]["behavior"].copy())
    hc=np.ones((n,2,3),dtype=np.float32);hc[:,0,0]=1+np.array([x["wording"] for x in rows])
    natural={"clean":dict(head_results=hc,behavior=data["clean"]["behavior"]),
             "corrupt":dict(head_results=np.zeros_like(hc),behavior=data["corrupt"]["behavior"])}
    selection=dict(families=[10,11],reference=fit_reference(raw,rows))
    result,metrics=analyze_secondary(selection,rows,data,natural,draws=50)
    assert result["recovery_fraction"]["wording"]["pooled"]["mean"]==.5
    np.testing.assert_allclose(result["recovery_fraction"]["self"]["pooled"]["mean"],.55)
    for ref in REFS:
        np.testing.assert_allclose(result["movement"]["wording"]["progress_toward_recipient"][ref]["pooled"]["mean"],.5)
        np.testing.assert_allclose(result["movement"]["wording"]["recipient_distance_closed"][ref]["pooled"]["mean"],.5)
        np.testing.assert_allclose(result["movement"]["wording"]["cosine_with_ideal_repair"][ref]["pooled"]["mean"],1.)
    assert result["paired_behavior_contrasts"]["wording_minus_residual_mean"]["logit"]["pooled"]["mean"]==1.
    # Independent scalar oracle for each inherited alternative distance.
    scale=np.array([[[2.,.5]],[[2.,.5]]]);e=data["wording"]["raw"][...,0];d=r[:,donors]
    centers=np.array([[[.5,2],[.5,2],[1.5,2],[1.5,2]]]*2)
    alt=alternative_margins(e,r,d,scale,centers,rows)
    for bi in range(2):
        for i in range(n):
            a,b,c0=e[bi,i],r[bi,i],d[bi,i];s=scale[bi,0]
            expected=(np.linalg.norm(a-b)-np.linalg.norm(a-c0))/np.linalg.norm(c0-b)
            np.testing.assert_allclose(alt["unscaled_paired_l2"][bi,i],expected)
            aa,bb,cc=a/s,b/s,c0/s
            expected=np.dot(aa,cc)/np.sqrt(np.dot(aa,aa)*np.dot(cc,cc))-np.dot(aa,bb)/np.sqrt(np.dot(aa,aa)*np.dot(bb,bb))
            np.testing.assert_allclose(alt["standardized_cosine"][bi,i],expected)
            expected=np.linalg.norm(aa-bb)-np.linalg.norm(aa-cc)
            np.testing.assert_allclose(alt["centroid_normalized_l2"][bi,i],expected)
    boot=Crossed(rows,draws=50);num=np.array([1+x["template"] for x in rows],float);den=np.array([2+8*x["template"] for x in rows],float)
    np.testing.assert_allclose(ratio_interval(boot,num,den)["mean"],.25)
    assert not ratio_interval(boot,num,np.zeros(n))["valid"]
    assert not pointwise(boot,np.full(n,np.nan))["valid"]
    json.dumps(result,allow_nan=False)
    return dict(status="passed",source="synthetic only",checks=["complete_synthetic_secondary_experiment",
        "three_distances_independent_scalar_oracles","half_repair_projection_and_distance",
        "paired_control_mean_contrast","ratio_of_means_not_mean_of_ratios",
        "zero_denominator_and_unresolved_values_retained","finite_json"],metric_arrays=len(metrics))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["freeze","self-test","final"],default="final")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.mode=="freeze":
        value=freeze();print(json.dumps(dict(freeze=str(FREEZE),utc=value["utc"])));return
    if args.mode=="self-test":
        value=self_test()
        if args.output:write(args.output,value)
        print(json.dumps(value));return
    completion=HERE/"results/inference-complete.json"
    assert completion.exists() and read(completion)["status"]=="completed", "Wait for complete final inference"
    verify_freeze()
    selection=read(HERE/"data/selection.json");rows=rows_for("test",selection["families"])
    folder=HERE/"results/test";paths=[HERE/"data/selection.json",HERE/"data/candidates.json",completion,FREEZE]
    def checked(name):
        path=folder/(name+".npz");record=read(folder/(name+"-record.json"))
        assert record["status"]=="completed" and record["ids"]==[r["id"] for r in rows]
        assert digest(path)==record["aggregate_sha256"],path
        paths.extend([path,folder/(name+"-record.json")])
        return load(path)
    data={name:checked(name) for name in CONDITIONS}
    natural={name:checked("natural-"+name) for name in ("clean","corrupt")}
    result,metrics=analyze_secondary(selection,rows,data,natural)
    result["input_hashes"]={str(p.relative_to(ROOT)):digest(p) for p in paths}
    result["secondary_analysis_freeze_sha256"]=digest(FREEZE)
    out=args.output or HERE/"results/secondary-analysis"
    assert not out.exists(),out
    out.mkdir(parents=True);write(out/"summary.json",result)
    np.savez_compressed(out/"per-example.npz",**metrics)
    print(json.dumps(dict(output=str(out),n=len(rows),scope=result["scope"])))


if __name__=="__main__":
    main()
