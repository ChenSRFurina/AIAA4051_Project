"""Validate the predeclared cumulative-budget selection, without model calls."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('part3_analysis',ROOT/'api/part3_analysis.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def checkpoint(round,calls,tokens,em):
    return {'round':round,'best_id':f'g{round}','cost':{'calls':calls,'generated_tokens':tokens},'em':em}


class MatchedBudgetTests(unittest.TestCase):
    def test_both_caps_apply(self):
        points=[checkpoint(0,61,3000,.7),checkpoint(1,163,9000,.8),checkpoint(2,265,20000,.9)]
        self.assertEqual(audit.choose_checkpoint(points,300,12000)['round'],1)
        self.assertEqual(audit.choose_checkpoint(points,200,30000)['round'],1)

    def test_latest_round_not_largest_score(self):
        points=[checkpoint(0,61,3000,.9),checkpoint(1,163,9000,.8)]
        self.assertEqual(audit.choose_checkpoint(points,200,10000)['round'],1)

    def test_caps_and_headroom_include_initialization(self):
        b=[checkpoint(0,61,3000,.7),checkpoint(1,163,9000,.8),checkpoint(2,265,12000,.9)]
        c=[checkpoint(0,61,3000,.7),checkpoint(1,163,16000,.8),checkpoint(2,265,23000,.9)]
        result=audit.matched_checkpoints(b,c,{'calls':265,'generated_tokens':12000},{'calls':265,'generated_tokens':23000})
        self.assertEqual(result['B']['round'],2)
        self.assertEqual(result['C']['round'],0)
        self.assertEqual(result['C']['unused_calls'],204)
        self.assertEqual(result['C']['unused_generated_tokens'],9000)

    def test_never_fill_with_incomplete_round(self):
        points=[checkpoint(0,61,3000,.7),checkpoint(1,163,9000,.8)]
        self.assertEqual(audit.choose_checkpoint(points,260,30000)['round'],1)

    def test_initialization_must_fit(self):
        with self.assertRaisesRegex(ValueError,'initialization'):
            audit.choose_checkpoint([checkpoint(0,61,3000,.7)],60,100000)


if __name__=='__main__':
    unittest.main()
