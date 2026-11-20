"""Sequential continuation of the active Part 3 run; never overlaps API jobs."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import time

from part12 import read,write
from run_part3 import ROOT,SESSION,BASELINES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ip')
    args=parser.parse_args()
    binding=['--source-ip',args.source_ip] if args.source_ip else []
    def status(stage,state='running',**extra):
        record={'stage':stage,'state':state,'time':datetime.now(timezone.utc).isoformat(),**extra}
        write(SESSION/'queue_status.json',record)
        print(record,flush=True)
    def command(stage,argv):
        status(stage)
        with (SESSION/(stage+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u',*argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status(stage,'failed',exit_code=result.returncode)
            raise SystemExit(result.returncode)
    status('waiting_for_searchqa_C')
    while not (SESSION/'searchqa/completed.json').exists():
        if (SESSION/'searchqa/interrupted.json').exists():
            status('searchqa_C','failed',reason=read(SESSION/'searchqa/interrupted.json')['error'])
            raise SystemExit(1)
        time.sleep(15)
    command('searchqa_audit',['api/part3_analysis.py','--baseline',str(BASELINES['searchqa']),'--experiment',str(SESSION/'searchqa'),'--output',str(SESSION/'searchqa/analysis.json')])
    # Freeze SearchQA C/prefix skills before running its test, independently of
    # HotpotQA outcomes. Freezing/auditing makes no generation requests.
    if not (SESSION/'searchqa/test_frozen.json').exists():
        command('searchqa_test_freeze',['api/test_part3.py','freeze',*binding])
    if not (SESSION/'hotpotqa/completed.json').exists():
        command('hotpotqa_C',['api/run_part3.py','hotpotqa',*binding])
    command('hotpotqa_audit',['api/part3_analysis.py','--baseline',str(BASELINES['hotpotqa']),'--experiment',str(SESSION/'hotpotqa'),'--output',str(SESSION/'hotpotqa/analysis.json')])
    if not (SESSION/'searchqa/test/results.json').exists():
        command('searchqa_C_test',['api/test_part3.py','test',*binding])
    status('experiments_finished','completed')


if __name__=='__main__':
    main()
