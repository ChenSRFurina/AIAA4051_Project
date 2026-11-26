"""Assemble the portable course evidence from immutable local evaluations (no inference)."""
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from part3_analysis import checked_rows, checkpoint_data
from report_hotpotqa_course import pairing
from baseline.io import digest, read_json, read_jsonl, write_json
from baseline.analysis import compare
from baseline.evaluation import qa_prompts
from baseline.budget import TokenCounter
from graphskillevo.envs.searchqa.evaluator import normalize_answer

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/final_project"
HP = ROOT / "Explore/runs/local/hotpotqa_course_20261124"
SQ = ROOT / "Explore/runs/local/searchqa_final_20261125"
OLD = ROOT / "Explore/runs/local/c2c-20261123-searchqa"
ARMS = ("N", "I", "B", "C", "C2b", "C2c")
LABELS = {
 "S": ("Answer form or span", "Possible same referent or value with a different name, inflection, phrasing or answer boundary."),
 "G": ("Qualifier or precision", "The answers differ in specificity, scope, unit, date resolution or a meaningful modifier."),
 "L": ("List membership", "One output includes additional entities or omits an entry or role from the reference."),
 "T": ("Wrong target or relation", "The prediction answers another slot, relation or intermediate entity."),
 "V": ("Different entity or value", "The reference and prediction identify different values; their divergence alone does not identify the failed hop."),
 "C": ("Comparison outcome", "The answer reverses, ties or fails to select between the compared alternatives."),
 "A": ("Abstention", "The response declines to provide the requested value, while the reference supplies one. Evidence sufficiency is a separate question."),
 "R": ("Question/reference concern", "The requested answer type, available evidence or wording of the reference raises an annotation question."),
 "F": ("Answer extraction", "A complete final answer tag is absent and the scorer uses fallback extraction."),
 "U": ("Unresolved", "The observed answer difference is real; the evidence does not resolve the cause.")
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired(left,right):
    p=compare(left,right)
    return {k:v for k,v in p.items() if k!="transitions"}


def compact(row):
    return {k:row[k] for k in ("id","question","gold_answers","predicted_answer","em","f1","format_ok",
                                "latency_seconds","visible_context_hash","skill_hash","context_truncated")}


def excerpt(row):
    """A literal, inspectable passage excerpt; no claim about which tokens the solver used."""
    lines=[x.strip() for x in row["visible_context"].splitlines() if x.strip() and not x.startswith("[DOC]")]
    if not lines:
        return ""
    gold=row["gold_answers"][0].lower()
    pred=row["predicted_answer"].lower()
    query=set(re.findall(r"\w{4,}",row["question"].lower()))
    def rank(s):
        low=s.lower()
        return (gold in low, bool(pred and pred in low),len(query.intersection(re.findall(r"\w{4,}",low))))
    line=max(lines,key=rank)
    return line[:700]


def diagnostic_details(row, no_skill, initial):
    """Describe observable answer differences without attributing a hidden reasoning step."""
    pred = normalize_answer(row["predicted_answer"]).split()
    ref = normalize_answer(row["gold_answers"][0]).split()
    def terms(tokens):
        return ", ".join(dict.fromkeys(tokens))[:110]
    missing = terms((Counter(ref)-Counter(pred)).elements())
    extra = terms((Counter(pred)-Counter(ref)).elements())
    parts = []
    if missing:
        parts.append("Reference-only tokens include " + missing + ".")
    if extra:
        parts.append("Prediction-only tokens include " + extra + ".")
    context = " " + normalize_answer(row["visible_context"]) + " "
    def present(tokens):
        return bool(tokens and (" " + " ".join(tokens) + " ") in context)
    pred_hit, ref_hit = present(pred), present(ref)
    if pred_hit and ref_hit:
        parts.append("Both answer strings occur in the supplied text after normalization.")
    elif ref_hit:
        parts.append("The normalized reference occurs as a contiguous span in the supplied text.")
    elif pred_hit:
        parts.append("The normalized prediction occurs as a contiguous span in the supplied text.")
    parts.append(f"Same-question EM: no skill {int(no_skill['em'])}, initial skill {int(initial['em'])}, evolved baseline 0.")
    return " ".join(parts)


def review(dataset, splits, labels):
    result=[]
    for split,arms in splits.items():
        rows=arms["B"]
        controls={a:{r["id"]:r for r in arms[a]} for a in ("N","I")}
        misses=[r for r in rows if not r["em"]]
        require(set(labels[split])=={r["id"] for r in misses},f"Review coverage mismatch: {dataset} {split}")
        for row in misses:
            entry=labels[split][row["id"]]
            require(entry[0]==digest(row["raw_response"]),"Review attached to another model response")
            code=entry[1]
            reason=(entry[2] if len(entry)>2 else
                    f'{LABELS[code][1]} Observed prediction “{row["predicted_answer"]}”; recorded reference “{row["gold_answers"][0]}”.')
            reason+=" "+diagnostic_details(row,controls["N"][row["id"]],controls["I"][row["id"]])
            result.append(dict(dataset=dataset,split=split,**compact(row),category=code,diagnosis=reason,
                               raw_response=row["raw_response"],context_excerpt=excerpt(row),response_sha256=entry[0]))
    return result


def candidate_evidence(run,init):
    points,records,rows=checkpoint_data(run,init)
    parent_map={op["child_id"]:op for h in read_json(run/"history.json") for op in h["operator_records"]}
    graphs={}
    for id,rec in records.items():
        graph=read_json(run/"skills"/(id+".json"))
        op=parent_map.get(id,{})
        parent=op.get("parent_ids",[])
        changes=[]
        if parent:
            for left,right in zip(rows[parent[0]],rows[id]):
                require(left["id"]==right["id"],"Candidate comparison order mismatch")
                if left["em"]!=right["em"]:
                    changes.append({"type":"repair" if right["em"] else "regression","id":right["id"],"question":right["question"],
                                    "before":left["predicted_answer"],"after":right["predicted_answer"],"gold":right["gold_answers"]})
        graphs[id]=dict(id=id,parent=parent,round=rec["generation"],operator=rec["origin"],text=rec["content"],
                        correct=round(rec["val_hard"]*len(rows[id])),changes=changes,**graph)
    return {"points":points,"graphs":graphs}


def physical(run):
    events=[e for e in read_jsonl(run/"ledger.jsonl") if e.get("physical_calls")]
    return {key:sum(e.get(key,0) for e in events) for key in ("physical_calls","prompt_tokens","generated_tokens","reasoning_tokens")} | {"failed":sum(e.get("status")!="ok" for e in events)}


def assemble():
    sqf=read_json(SQ/"frozen.json")
    sqr=read_json(SQ/"completed.json")
    require(digest(sqf)==sqr["freeze_id"],"Wrong SearchQA freeze")
    hp_summary=read_json(ROOT/"results/hotpotqa_course/summary.json")
    hpf=read_json(HP/"frozen.json")
    require(hpf["freeze_id"]==hp_summary["freeze_id"],"Wrong HotpotQA freeze")
    hp_matches=hp_summary["matched"]["arms"]
    config=read_json(HP/"B/config.json")
    sqconfig=sqf["config"]
    for field in ("id","revision","precision","server_version","context_window","extra_body","structured_operators"):
        require(config["model"][field]==sqconfig["model"][field],"Model mismatch: "+field)
    for f in ("qa_temperature","qa_max_tokens","operator_max_tokens","seed","top_p","operator_temperature"):
        require(config[f]==sqconfig[f],"Sampling mismatch: "+f)
    require(hpf["signature"]["tokenizer_hash"]==read_json(SQ/"policy.json")["tokenizer_hash"],"Tokenizer mismatch")
    metadata=read_json(ROOT/"local_model/deployment_validation.json")["model_manifest"]
    summary={"model":config["model"],"gguf_sha256":metadata["gguf_sha256"],"datasets":{}}
    write_json(DEST/"protocols.json", {"hotpotqa":read_json(HP/"policy.json"),"searchqa":read_json(SQ/"policy.json"),
               "hotpotqa_config":config, "searchqa_config":sqconfig})
    observations, evolution, inventory, batch_ids, prompts={},{},[],{},{}
    request_examples={}
    for dataset,run,init in (("hotpotqa",HP,HP/"shared_init"),("searchqa",SQ,Path(read_json(Path(sqf["paths"]["B"])/"shared_initialization.json")["path"]))):
        data=ROOT/"Baseline/data/prepared"/("hotpotqa_course" if dataset=="hotpotqa" else "searchqa")
        ids={k:read_json(data/(k+"_ids.json")) for k in ("train","dev","test")}
        batch_ids[dataset]=ids
        paths={a:(run/a if dataset=="hotpotqa" else Path(sqf["paths"][a])) for a in ARMS}
        examples = {}
        for directory in (init,paths["B"]):
            with (directory/"calls.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    call=json.loads(line)
                    if call["stage"] not in examples:
                        examples[call["stage"]]=call["request"]
        request_examples[dataset]=examples
        all_test,all_dev={},{}
        graphs,stats,material={}, {}, {}
        ledgers=[physical(init)]
        for a in ARMS:
            path=paths[a]
            meta=read_json(path/"summary.json")
            source=sqr["results"][a]["source"] if dataset=="searchqa" else a
            test=run/"test"/source
            dr=read_json(path/"final_dev_outputs.json")
            tr=read_json(test/"outputs.json")
            for k,rr in (("dev",dr),("test",tr)):
                require([r["id"] for r in rr]==ids[k],dataset+" "+k+" IDs changed")
            skill=(path/"best_skill.md").read_text(encoding="utf8")
            require(all(r["skill_hash"]==digest(skill) and not r["error"] for r in dr+tr),"Invalid selected outputs")
            all_test[a],all_dev[a]=tr,dr
            stats[a]=dict(dev=checked_rows(dr,dataset),test=checked_rows(tr,dataset),best=meta["best"]["id"],
                          skill_hash=digest(skill),logical_dev_cost=meta["cost"],test_source=source,
                          test_new_cost=physical(test) if a==source else {"physical_calls":0})
            ledgers.append(physical(path))
            if a not in ("N","I"):
                graphs[a]=candidate_evidence(path,init)
                material[a]={}
                for op in sorted((path/"operators").glob("*.json")):
                    rec=read_json(op)
                    material[a][op.stem]=[{"status":at["status"],"prompt":at.get("prompt"),"error":at.get("error")} for at in rec["attempts"]]
        for folder in (run/"test").iterdir():
            if folder.is_dir() and (folder/"ledger.jsonl").exists(): ledgers.append(physical(folder))
        if dataset=="hotpotqa":
            # Both matched checkpoints produce B's exact prompt for every test question.
            tcounter=TokenCounter(config["model"]["tokenizer_path"])
            for a in ("B","C"):
                m=hp_matches[a]
                item=graphs[a]["graphs"][m["best_id"]]
                require(digest(item["text"])==m["skill_hash"]==stats["B"]["skill_hash"],"Matched HotpotQA skill mismatch")
                for row in all_test["B"]:
                    directory=run/"test/B/predictions"/row["id"]
                    system,user,visible=qa_prompts({"question":row["question"],"context":row["visible_context"]},item["text"],config=config,counter=tcounter)
                    require(system==(directory/"system.txt").read_text(encoding="utf8") and user==(directory/"user.txt").read_text(encoding="utf8")
                            and digest(visible)==row["visible_context_hash"],"Matched HotpotQA prompt mismatch")
            matched={a:dict(checkpoint=hp_matches[a],test_source="B",metrics=stats["B"]["test"],extra_test_calls=0) for a in ("B","C")}
            cap={k:hp_summary["matched"][k] for k in ("calls","tokens")}
        else:
            matched={a:dict(checkpoint=sqf["matched"][a],test_source=sqr["results"]["matched_"+a]["source"],
                           metrics=sqr["results"]["matched_"+a]["metrics"],extra_test_calls=sqr["results"]["matched_"+a]["new_calls"]) for a in ("B","C")}
            cap={"calls":sqf["caps"]["calls"],"tokens":sqf["caps"]["generated_tokens"]}
        labels=read_json(DEST/("review_labels.json" if dataset=="hotpotqa" else "searchqa_review_labels.json"))
        inventory.extend(review(dataset,{"dev":all_dev,"test":all_test},labels))
        observations[dataset]={split:{a:[compact(r) for r in by_arm[a]] for a in ARMS} for split,by_arm in (("dev",all_dev),("test",all_test))}
        evolution[dataset]=graphs
        prompts[dataset]=material
        summary["datasets"][dataset]={"arms":stats, "pairing":{a:pairing(paths[a]) for a in ("C","C2b","C2c")},
                    "compared_to_B":{a:paired(all_test["B"],all_test[a]) for a in ("N","I","C","C2b","C2c")},
                    "matched":matched,"caps":cap,"ids_sha256":digest(ids),"freeze_id":sqr["freeze_id"] if dataset=="searchqa" else hpf["freeze_id"],
                    "physical":{k:sum(l[k] for l in ledgers) for k in ledgers[0]},"warm_start_cost":physical(init)}
    # Select demo material by a known mutation and retain exact recorded text.
    op=read_json(HP/"C2c/operators/g0003_i0000.json")
    packet=read_json(HP/"C2c/evaluations/g3/g0001_i0000/train/feedback.json")
    demo={"mutation":op["operator"],"parent":op["parent_ids"][0],"child":op["child_id"],"packet":next(p for p in packet["actual_packets"] if p["failure_id"]=="5a81e20e55429926c1cdadae"),
          "feedback":packet["actual_feedback"],"adopted_prompt":next(a["prompt"] for a in op["attempts"] if a["status"]=="ok")}
    require(demo["feedback"] in demo["adopted_prompt"],"Demo feedback not found in operator input")
    write_json(DEST/"demo_evidence.json",demo)
    events=read_jsonl(ROOT/"Baseline/runs/gpu_accounting.jsonl")
    groups={}
    for e in events: groups.setdefault(e["session"],{})[e["action"]]=e
    spans=[]
    for name,pair in groups.items():
        require(set(pair)=={"start","stop"},f"Incomplete service allocation {name}")
        b,e=pair["start"],pair["stop"]
        spans.append(dict(session=name,start=b["utc"],end=e["utc"],hours=(datetime.fromisoformat(e["utc"])-datetime.fromisoformat(b["utc"])).total_seconds()/3600*b["gpu_count"]))
    summary["gpu"]={"local_spans":spans,"local_hours":sum(r["hours"] for r in spans),"team_api_gpu_hours":None,"proposed_cap":40}
    summary["taxonomy"]=LABELS
    summary["failure_counts"]={ds:{sp:dict(Counter(r["category"] for r in inventory if r["dataset"]==ds and r["split"]==sp)) for sp in ("dev","test")} for ds in summary["datasets"]}
    for name,obj in (("summary",summary),("predictions",observations),("skill_graphs",evolution),("failure_inventory",inventory),
                     ("fixed_ids",batch_ids),("operator_inputs",prompts),("request_examples",request_examples),("gpu_log",events)):
        write_json(DEST/(name+".json"),obj)
    with (DEST/"failure_inventory.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["dataset","split","id","question","gold_answers","predicted_answer","em","f1","category","diagnosis","context_excerpt","raw_response","response_sha256"]
        writer=csv.DictWriter(f,fields,extrasaction="ignore");writer.writeheader()
        for row in inventory:writer.writerow({**row,"gold_answers":json.dumps(row["gold_answers"],ensure_ascii=False)})
    print("Verified",len(inventory),"baseline failures, both datasets, all final arms and both matched-budget tests.")


if __name__=="__main__":
    assemble()
