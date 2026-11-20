"""Compact, read-only status for the fixed C2b session."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1] / 'Explore/runs/api/c2b-20261110'


def status():
    for dataset in ['searchqa', 'hotpotqa']:
        run = ROOT / dataset / 'C2b'
        ledger = run / 'ledger.jsonl'
        if not ledger.exists():
            print(json.dumps({'dataset': dataset, 'status': 'not_started'}))
            continue
        # A writer may be appending the last line; parse only complete lines.
        contents = ledger.read_text(encoding='utf-8')
        lines = contents.splitlines()
        if contents and not contents.endswith('\n'): lines = lines[:-1]
        events = [json.loads(line) for line in lines if line.strip()]
        physical = [e for e in events if e.get('physical_calls')]
        result = {'dataset': dataset, 'physical_calls': len(physical),
                  'logical_calls': sum(e.get('calls', 0) for e in events),
                  'generated_tokens': sum(e.get('generated_tokens', 0) for e in events),
                  'phase_counts': dict(Counter(e.get('phase') for e in physical)),
                  'failed_requests': sum(e.get('status') != 'ok' for e in physical),
                  'last_phase': physical[-1].get('phase') if physical else None}
        if (run / 'summary.json').exists():
            summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
            result.update(status=summary['status'], rounds=summary['completed_rounds'],
                          dev_correct=summary['dev']['correct'], best=summary['best']['id'])
        else:
            result['status'] = 'running_before_first_checkpoint'
        if (run.parent / 'interrupted.json').exists():
            result['interruption'] = json.loads((run.parent / 'interrupted.json').read_text(encoding='utf-8'))['error']
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    status()
