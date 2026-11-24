"""Regression checks for the course HotpotQA split and full-context QA policy."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Baseline"))
from prepare_hotpotqa_course import LEVELS, select_ids
from baseline.evaluation import qa_prompts


class CourseHotpotQATest(unittest.TestCase):
    def test_split_ids_are_fixed_disjoint_stratified_and_skip_old_dev(self):
        pools = {"train": {f"{lev}{i}": lev for lev in LEVELS for i in range(170)},
                 "validation": {f"official_dev_{i}": "hard" for i in range(1050)}}
        old = list(pools["validation"])[:30]
        ids = select_ids(pools, old)
        reversed_source = {key: dict(reversed(list(rows.items()))) for key, rows in pools.items()}
        self.assertEqual(ids, select_ids(reversed_source, old[::-1]))
        self.assertEqual([len(ids[k]) for k in ("train_ids","val_ids","test_ids")], [300,150,1000])
        self.assertEqual(len(set().union(*ids.values())), 1450)
        self.assertTrue(set(old).isdisjoint(ids["test_ids"]))
        for lev in LEVELS:
            self.assertEqual(sum(pools["train"][i] == lev for i in ids["train_ids"]), 100)
            self.assertEqual(sum(pools["train"][i] == lev for i in ids["val_ids"]), 50)

    def test_course_solver_receives_last_paragraph_and_no_scoring_metadata(self):
        context = "[DOC] First\n" + "text " * 2500 + "\n[DOC] Last\nrequired final sentence"
        item = {"context": context, "question": "Which paragraph?", "answers": ["REFERENCE_ONLY"],
                "metadata": {"supporting_facts": ["GOLD_LABEL_ONLY"]}}
        system, user, visible = qa_prompts(item, "Use the documents.", config={"course_hotpotqa": True})
        self.assertEqual(visible, context)
        self.assertIn("required final sentence", user)
        self.assertNotIn("REFERENCE_ONLY", system + user)
        self.assertNotIn("GOLD_LABEL_ONLY", system + user)


if __name__ == "__main__":
    unittest.main()

