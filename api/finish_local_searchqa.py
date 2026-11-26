"""Freeze the local SearchQA comparison and test exact skills under a common budget."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/20261110/protocols/protocol-067f9db47e18"
sys.path[:0] = [str(ARCHIVE / "Baseline"), str(ARCHIVE / "Explore"), str(ROOT / "Explore")]
from baseline.io import read_json, write_json, digest
from baseline.trainer import CourseTrainer, tree_fingerprint
from baseline.data import load_data
from baseline.config import config_hash
from baseline.protocol import artifact_hashes, freeze_design, check_design
from baseline.budget import Budget, TokenCounter
from baseline.client import ChatClient
from baseline.evaluation import run_qa, aggregate
from explore.feedback import ContrastiveFeedback
from explore_v2.feedback import DiagnosticFeedback
from explore_c2c import ClueFeedback
from part3_analysis import checkpoint_data, choose_checkpoint, checked_rows

OLD = ROOT / "Explore/runs/local/c2c-20261123-searchqa"
OUT = ROOT / "Explore/runs/local/searchqa_final_20261125"
DATA = ROOT / "Baseline/data/prepared/searchqa"
ARMS = ("N", "I", "B", "C", "C2b", "C2c")


def seal(path, obj):
    if path.exists() and read_json(path) != obj:
        raise ValueError(f"Frozen inputs changed: {path}")
    if not path.exists():
        write_json(path, obj)


def run(out, reuse):
    public = read_json(ROOT / "results/20261123/design.json")
    config = dict(public["config"])
    if reuse is None:
        config["gpu_ledger_path"] = str(ROOT / "Baseline/runs/gpu_accounting.jsonl")
        config["model"] = dict(config["model"], tokenizer_path=str(ROOT / "api/runtime/tokenizer"))
    protocol = tree_fingerprint()
    if protocol != public["protocol"] or any(digest((ROOT / f).read_bytes()) != h for f,h in public["files"].items()):
        raise ValueError("Archived protocol or Explore implementation changed")
    splits, manifest = load_data(DATA, "searchqa")
    counter = TokenCounter(config["model"]["tokenizer_path"])
    signature = dict(config_hash=config_hash(config), data_hash=digest(manifest), code_hash=protocol, tokenizer_hash=counter.identity)
    policy = dict(signature, seed=config["seed"], arms=list(ARMS), primary=["B","C"],
                  rule="For the additional matched B/C comparison take the last complete round under both minimum final logical-call and generated-token totals. Reuse strictly identical test skills.",
                  exposure="SearchQA test was seen in the earlier hosted experiment. No method changes or selection using those outputs.",
                  ids={s:[r["id"] for r in items] for s,items in splits.items()}, driver=digest(Path(__file__).read_bytes()))
    seal(out/"policy.json",policy)
    init = (reuse or out) / "shared_init"
    if reuse is None and not (init/"summary.json").exists():
        CourseTrainer(config,DATA,init,condition="shared_initialization").initialize()
    if any(read_json(init/"summary.json")[k] != v for k,v in signature.items()):
        raise ValueError("Shared initialization signature mismatch")
    paths = {}
    methods = {"C":ContrastiveFeedback,"C2b":DiagnosticFeedback,"C2c":ClueFeedback}
    for a in ARMS:
        path = ((reuse or out) if a in ("B","C2b","C2c") else out)/a
        paths[a] = path
        if a=="C":
            if not (out/"feedback_design.json").exists():
                freeze_design(paths["B"],out/"feedback_design.json",config)
            check_design(out/"feedback_design.json",config)
        if not (path/"summary.json").exists():
            if reuse and a in ("B","C2b","C2c"):
                raise ValueError("Missing declared reused development run")
            print("Development",a,flush=True)
            feedback=methods[a]("searchqa",policy["ids"]["train"],path,config) if a in methods else None
            tr=CourseTrainer(config,DATA,path,condition=a,feedback=feedback)
            if a in ("N","I"): tr.static()
            else: tr.train_course(init)
        summary=read_json(path/"summary.json")
        if any(summary[k]!=v for k,v in signature.items()) or summary["status"] not in {"completed","static_complete"} or summary["dev"]["n"]!=30:
            raise ValueError(f"Invalid development evidence for {a}")
        if a not in ("N","I") and (summary["completed_rounds"]!=3 or read_json(path/"shared_initialization.json")["population_hash"]!=digest(read_json(init/"population.json"))):
            raise ValueError(f"Shared population mismatch for {a}")
    skills={a:(p/"best_skill.md").read_text(encoding="utf-8") for a,p in paths.items()}
    points={a:checkpoint_data(paths[a],init) for a in ("B","C")}
    caps={k:min(read_json(paths[a]/"summary.json")["cost"][k] for a in ("B","C")) for k in ("calls","generated_tokens")}
    matched={a:choose_checkpoint(points[a][0],caps["calls"],caps["generated_tokens"]) for a in ("B","C")}
    for a,m in matched.items(): skills["matched_"+a]=points[a][1][m["best_id"]]["content"]
    frozen={"policy":digest(policy),"config":config,"paths":{k:str(v) for k,v in paths.items()},
            "artifact_hashes":{k:artifact_hashes(p) for k,p in paths.items()},
            "skills":{k:digest(v) for k,v in skills.items()},"caps":caps,"matched":matched}
    seal(out/"frozen.json",frozen)
    print("Frozen",digest(frozen),flush=True)
    previous, results = {}, {}
    for a,skill in skills.items():
        key=digest(skill)
        # Primary N/I/B/C have their own physical test observation.
        source=previous.get(key) if a not in ("N","I","B","C") else None
        dest=out/"test"/a
        if source:
            results[a]=dict(results[source], source=source, new_calls=0)
            continue
        budget=Budget(dest/"ledger.jsonl",config["budget"])
        client=ChatClient(config,budget,counter,dest/"calls.jsonl")
        rows=[]
        for item in splits["test"]:
            ck=dest/"predictions"/item["id"]/"result.json"
            row=read_json(ck) if ck.exists() else None
            if row is None or row["error"]:
                row=run_qa(client,item,skill,split="test",round_no=-1,seed=config["seed"],condition=a,out_dir=dest)
            if (row["id"],row["skill_hash"],row["condition"],row["gold_answers"])!=(item["id"],key,a,item["answers"]):
                raise ValueError("Wrong frozen test checkpoint")
            rows.append(row)
        metrics=checked_rows(rows,"searchqa")
        write_json(dest/"outputs.json",rows)
        previous.setdefault(key,a)
        results[a]={"source":a,"metrics":metrics,"new_calls":budget.totals()["physical_calls"],"cost":budget.totals()}
        print(a,metrics["correct"],"/",metrics["n"],flush=True)
    write_json(out/"completed.json",dict(freeze_id=digest(frozen),results=results))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--fresh",action="store_true",help="Rebuild all development arms without the prior local runs.")
    args=parser.parse_args()
    run(args.output.resolve(),None if args.fresh else OLD)
