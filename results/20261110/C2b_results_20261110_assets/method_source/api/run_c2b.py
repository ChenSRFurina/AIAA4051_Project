"""C2b train/dev only, injected into each unchanged archived B/C protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import part12
from run_part3 import ROOT, BASELINES, setup, sha

SESSION = ROOT / 'Explore/runs/api/c2b-20261110'


def method_files():
    files = sorted((ROOT / 'Explore/explore_v2').glob('*.py')) + [Path(__file__)]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files}


def train(dataset, output, source_ip=None):
    p, config, baseline, snapshot, _ = setup(dataset, source_ip)
    output = Path(output).resolve()
    p.ensure_empty(output)
    hashes = method_files()
    for relative in hashes:
        target = output / 'method_source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    sys.path.insert(0, str(output / 'method_source/Explore'))
    from explore_v2.feedback import DiagnosticFeedback
    from baseline.trainer import CourseTrainer
    policy = {
        'method': 'C2b_diagnostic_evidence_feedback_v1', 'dataset': dataset,
        'created_at': datetime.now(timezone.utc).isoformat(), 'baseline': str(baseline),
        'protocol_root': str(snapshot), 'protocol_code_hash': p.tree_fingerprint(),
        'config_hash': p.config_hash(config), 'method_files': hashes, 'method_hash': p.digest(hashes),
        'source_ip': source_ip, 'python': sys.executable,
        'scope': 'train_dev_only_no_test_evaluation',
        'changes': ['train diagnostic categories', 'relevant exact evidence spans with provenance',
                    'conditional executable and operator-scope instructions',
                    'diagnostic failure-only input for unpaired cases instead of verbatim B fallback'],
        'unchanged': ['pair_failure and threshold', 'own train memory and capacity', 'random k=3 failure selection',
                      'shared initialization', 'QA prompts and decoding', 'upstream survivor/final selection and tie handling',
                      '3 rounds x 2 candidates', 'operator retry cap', 'all config and budget limits'],
        'no_new_judge_calls': True, 'semantic_audits_are_read_only': True,
        'comparison': 'Historical B and C, exact shared initialization/config/protocol per dataset; not contemporaneous reruns. Hosted model revision remains undisclosed.',
        'matched_budget': 'Secondary: minimum final logical calls and generated tokens across B,C,C2b; latest complete round within both ceilings; no test-based choice.',
        'train_metrics': 'Parent rollouts on fixed per-round train batches; not a full-train final-model evaluation.',
        'selection': 'Unchanged archived validation EM rules; dev is used for selection and cannot establish held-out generalization.'
    }
    policy['policy_id'] = p.digest(policy)
    part12.write(output / 'policy.json', policy)
    train_ids = part12.read(ROOT / 'Baseline/data/prepared' / dataset / 'train_ids.json')
    feedback = DiagnosticFeedback(dataset, train_ids, output / 'C2b', config)
    trainer = CourseTrainer(config, ROOT / 'Baseline/data/prepared' / dataset,
                            output / 'C2b', condition='C2b', feedback=feedback)
    part12.write(output / 'C2b/method.json', {'method_hash': policy['method_hash'],
                  'protocol_code_hash': policy['protocol_code_hash'], 'policy_id': policy['policy_id'],
                  'note': 'run_meta.code_hash identifies the unchanged shared protocol; method_hash additionally identifies C2b.'})
    # Progress only: no change to parent sampling, calls, scoring or selection.
    original_save = trainer.save_final
    def save(population, status):
        result = original_save(population, status)
        print(json.dumps({'dataset': dataset, 'status': status, 'rounds': result['completed_rounds'],
                          'dev': result['dev'], 'cost': result['cost']}, ensure_ascii=False), flush=True)
        return result
    trainer.save_final = save
    print(json.dumps({'status': 'starting_C2b', 'dataset': dataset, 'output': str(output),
                      'method_hash': policy['method_hash'], 'budget': config['budget']}), flush=True)
    try:
        result = trainer.train_course(baseline / 'shared_init')
        part12.require(result['status'] == 'completed' and result['completed_rounds'] == 3, 'Incomplete C2b run')
        part12.require(method_files() == hashes, 'C2b source changed during execution')
        part12.require(p.tree_fingerprint() == policy['protocol_code_hash'], 'Shared protocol changed')
        part12.require(part12.read(baseline / 'B/shared_initialization.json')['population_hash'] ==
                       part12.read(output / 'C2b/shared_initialization.json')['population_hash'], 'Initialization mismatch')
        part12.write(output / 'completed.json', {'status': 'completed', 'dataset': dataset,
                      'policy_id': policy['policy_id'], 'method_hash': policy['method_hash'],
                      'dev': result['dev'], 'cost': result['cost'], 'finished_at': datetime.now(timezone.utc).isoformat()})
        return result
    except Exception as exc:
        part12.write(output / 'interrupted.json', {'error': str(exc), 'policy_id': policy['policy_id'],
                      'time': datetime.now(timezone.utc).isoformat(), 'outputs_preserved': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['searchqa', 'hotpotqa'])
    parser.add_argument('--output')
    parser.add_argument('--source-ip')
    args = parser.parse_args()
    train(args.dataset, args.output or SESSION / args.dataset, args.source_ip)
