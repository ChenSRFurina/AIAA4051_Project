"""Run the sealed Option-A method against its exact baseline protocol snapshot.

Only orchestration, auditing and reporting are new. No frozen optimizer inputs,
failure sampling, model settings, initialization or selection rules are modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import part12

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {
    "searchqa": ROOT / "Baseline/runs/api/searchqa/20261107-125241-971323-38572",
    "hotpotqa": ROOT / "Baseline/runs/api/hotpotqa/20261109-153042-871031-7652",
}
PRIOR_SEAL = ROOT / "Baseline/runs/part12/searchqa/frozen.json"
SESSION = ROOT / "Explore/runs/api/part3-20261110"
POLICY = {
    "method": "pre_test_sealed_success_failure_contrastive",
    "method_changes": False,
    "baseline_limits": "exact same frozen config, calls and generation-token caps",
    "main_comparison": "final validation-selected skills after at most 3 rounds / 2 new candidates per round",
    "matched_prefix": "Set a common rectangle of min(B final calls,C final calls) and min(B final generated tokens,C final generated tokens), including shared initialization once per logical method. Select each method's latest COMPLETE round inside BOTH caps (initialization is round 0). Use that checkpoint's validation-selected best, never test scores. Partial rounds are excluded. Report unused headroom. This is an accounting-based secondary analysis, not exact equal realized-token consumption.",
    "pairing_denominator": "selected failures of actual mutation parents; reused parent feedback counted once per operator exposure, plus unique parent-feedback counts separately",
    "test_rule": "SearchQA existing N/I/B test observations are reused with their freeze provenance. Execute C test only after its final and matched-prefix choices are frozen. If the prefix selected skill differs from an already evaluated condition, evaluate it separately with the same limits and label secondary. HotpotQA test requires course labels/partition.",
    "test_noninterference": "No test outputs, answers, reviews or scores are used to change C or to select any checkpoint.",
    "limitations": "Baseline test was exposed before this run, but C method/config were sealed beforehand. Service weight revision/precision unknown. Public HotpotQA is development only.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(dataset, source_ip=None, baseline=None, prior_seal=None):
    baseline = Path(baseline or BASELINES[dataset]).resolve()
    bmeta = part12.read(baseline / "B/run_meta.json")
    snapshot = ROOT / "api/runtime" / ("protocol-" + bmeta["code_hash"][:12])
    p = part12.bootstrap(snapshot)
    part12.require(p.tree_fingerprint() == bmeta["code_hash"], "Training protocol snapshot changed")
    config = part12.read(baseline / "B/config.json")
    part12.require(config["dataset"] == dataset and p.config_hash(config) == bmeta["config_hash"], "Baseline config mismatch")
    # Seal is an immutable pre-test method record, not a source of test observations.
    seal = part12.read(prior_seal or PRIOR_SEAL)
    part12.require(seal["freeze_id"] == p.digest({k:v for k,v in seal.items() if k!="freeze_id"}), "Prior seal modified")
    for name, expected in seal["future_C"]["files"].items():
        part12.require(sha(snapshot / name) == expected, "C differs from its pre-test implementation seal")
        part12.require(sha(ROOT / name) == expected, "Working C differs from its pre-test implementation seal")
    if dataset == "searchqa":
        part12.require(config == seal["config"], "SearchQA config differs from pre-test seal")
        for c in ["N","I","B"]:
            part12.require(p.artifact_hashes(baseline/c) == seal["conditions"][c]["artifacts"], "Baseline evidence modified since test freeze")
    else:
        for c in ["N","I","B"]:
            meta = part12.read(baseline/c/"run_meta.json")
            part12.require(all(meta[k] == bmeta[k] for k in ["config_hash","data_hash","code_hash","tokenizer_hash"]), "HotpotQA baseline signature mismatch")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False, interpolate=False)
    if source_ip:
        part12.install_source_transport(source_ip,config["model"]["base_url"])
    sys.path.insert(0,str(snapshot/"Explore"))
    return p,config,baseline,snapshot,seal


def train(dataset, output, source_ip=None):
    p,config,baseline,snapshot,seal=setup(dataset,source_ip)
    output=Path(output).resolve()
    p.ensure_empty(output)
    from explore.feedback import ContrastiveFeedback
    from baseline.trainer import CourseTrainer
    from baseline.protocol import freeze_design,check_design
    data=ROOT/"Baseline/data/prepared"/dataset
    train_ids=part12.read(data/"train_ids.json")
    policy={**POLICY,"dataset":dataset,"created_at":datetime.now(timezone.utc).isoformat(),
            "baseline":str(baseline),"protocol_root":str(snapshot),"driver_sha256":sha(__file__),
            "prior_method_seal":seal["freeze_id"],"code_hash":p.tree_fingerprint(),
            "config_hash":p.config_hash(config),"source_ip":source_ip,"python":sys.executable}
    policy["policy_id"]=p.digest(policy)
    part12.write(output/"policy.json",policy)
    freeze_design(baseline/"B",output/"feedback_design.json",config)
    design=check_design(output/"feedback_design.json",config)
    feedback=ContrastiveFeedback(dataset,train_ids,output/"C",config)
    trainer=CourseTrainer(config,data,output/"C",condition="C",feedback=feedback)
    part12.write(output/"C/feedback_design.json",design)
    print(json.dumps({"status":"starting_sealed_C","dataset":dataset,"output":str(output),"code_hash":p.tree_fingerprint()}),flush=True)
    try:
        result=trainer.train_course(baseline/"shared_init")
        part12.require(result["status"]=="completed" and result["completed_rounds"]==3,"C did not complete the configured three rounds")
        b_init=part12.read(baseline/"B/shared_initialization.json")
        c_init=part12.read(output/"C/shared_initialization.json")
        part12.require(b_init["population_hash"]==c_init["population_hash"],"B/C initialization mismatch")
        part12.require(p.tree_fingerprint()==policy["code_hash"],"Protocol changed during training")
        part12.write(output/"completed.json",{"status":"completed","dataset":dataset,"policy_id":policy["policy_id"],"dev":result["dev"],"cost":result["cost"],"finished_at":datetime.now(timezone.utc).isoformat()})
        return {"status":"completed","dataset":dataset,"dev":result["dev"],"cost":result["cost"]}
    except Exception as exc:
        part12.write(output/"interrupted.json",{"error":str(exc),"time":datetime.now(timezone.utc).isoformat(),"policy_id":policy["policy_id"],"outputs_preserved":True})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',choices=['searchqa','hotpotqa'])
    parser.add_argument('--output')
    parser.add_argument('--source-ip')
    args=parser.parse_args()
    result=train(args.dataset,args.output or SESSION/args.dataset,args.source_ip)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
