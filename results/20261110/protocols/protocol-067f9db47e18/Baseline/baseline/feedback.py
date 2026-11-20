"""Unchanged upstream random failure selection and baseline reflection formatting."""
from graphskillevo.evaluation import _sample_failure_rows, build_rollout_reflection_context


def selected_failures(rows, k, seed):
    return _sample_failure_rows(rows, failure_k=k, seed=seed)


def fit_parts(counter, header, parts, maximum):
    """Give each selected failure the same allowance; preserve the selected-ID log."""
    header = counter.clip(header, min(counter.count(header), maximum // 2))
    if not parts:
        return header
    allowance = max(0, (maximum - counter.count(header) - 8 * len(parts)) // len(parts))
    result = header + "\n\n" + "\n\n---\n\n".join(counter.clip(part, allowance) for part in parts)
    return counter.clip(result, maximum)


class BaselineFeedback:
    name = "failure_only"

    def observe(self, rows):
        return

    def build(self, rows, items, out_dir, seed, config, counter):
        failures = selected_failures(rows, config["failure_k"], seed)
        class Reference:
            def build_reference_text(self, item):
                return "Gold answers: " + repr(item["answers"])
        parts = [build_rollout_reflection_context([row], str(out_dir), source_items=items, adapter=Reference(),
                                                  failure_k=1, seed=seed, include_hidden_reference=True) for row in failures]
        text = fit_parts(counter, "", parts, config["feedback_max_tokens"]) if parts else ""
        return text, {"mode": self.name, "selected_failure_ids": [r["id"] for r in failures],
                      "feedback_tokens": counter.count(text), "same_question": 0, "similar_task": 0, "unpaired": len(failures)}
