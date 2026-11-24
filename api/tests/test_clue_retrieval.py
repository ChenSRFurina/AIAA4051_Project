import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "Baseline"), str(ROOT / "Explore")]
from explore_c2c import retrieve, target_kind


class Memory:
    def __init__(self, rows):
        self.rows = rows

    def successes(self):
        return self.rows


def row(key, question, generation=1):
    return {"id":key,"question":question,"dataset":"searchqa","split":"train","round":generation,
            "format_ok":True,"visible_context_hash":key,"skill_hash":"abc","observation_id":key}


class ClueRetrievalTests(unittest.TestCase):
    def test_same_question_has_priority_even_if_used(self):
        failure = row("f","This author wrote a story")
        ref = row("f",failure["question"])
        unrelated = row("p","This artist painted an ocean")
        result, rule = retrieve(failure,Memory([unrelated,ref]),0.2,{"f"})
        self.assertEqual((rule,result["pair_type"],result["success_reference"]["id"]),("original","same_question","f"))

    def test_unknown_future_and_nonmatching_targets_abstain(self):
        failure = row("f","This author wrote a story")
        options = [row("future","She composed a symphony",2),row("place","This city was rebuilt")]
        result, rule = retrieve(failure, Memory(options),0.2,set())
        self.assertIsNone(result["success_reference"])
        self.assertEqual(rule,"unpaired")
        self.assertIsNone(target_kind("Acorns and rain"))

    def test_fallback_prefers_a_fresh_reference_with_the_same_target(self):
        failure = row("f","He designed a mechanism")
        best_lexically = row("repeat","This inventor designed a mechanism")
        unused = row("fresh","She composed several symphonies")
        result,rule = retrieve(failure,Memory([unused,best_lexically]),0.2,{"repeat"})
        self.assertEqual(rule,"target_kind:person")
        self.assertEqual(result["success_reference"]["id"],"fresh")


if __name__ == "__main__":
    unittest.main()
