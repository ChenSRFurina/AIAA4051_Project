"""Train-only, per-condition memory. Raw logs remain uncompressed."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from baseline.io import digest


class TrainMemory:
    def __init__(self, dataset, allowed_ids, condition="C", per_outcome_limit=2):
        self.dataset, self.allowed_ids, self.condition = dataset, set(allowed_ids), condition
        self.limit = per_outcome_limit
        self._groups = defaultdict(list)

    def add(self, row):
        if row["split"] != "train" or row["dataset"] != self.dataset or row["id"] not in self.allowed_ids:
            raise ValueError("Only the frozen train partition may enter memory")
        if row["condition"] not in {self.condition, "shared_initialization"}:
            raise ValueError("Foreign method trajectories cannot enter this method's memory")
        if row.get("cached") or row.get("error"):
            return False
        key = (row["id"], row["hard"])
        identity = digest([row["raw_response"], row["visible_context_hash"]])
        group = self._groups[key]
        for previous in group:
            if previous["dedup_hash"] == identity:
                # Update provenance to a true newly executed latest observation, never count duplicates twice.
                if row["round"] > previous["round"]:
                    group.remove(previous)
                    break
                return False
        entry = deepcopy(row)
        entry["dedup_hash"] = identity
        group.append(entry)
        group.sort(key=lambda r: (-r["round"], r["skill_hash"], r["observation_id"]))
        del group[self.limit:]
        return True

    def records(self):
        return [deepcopy(row) for key in sorted(self._groups) for row in self._groups[key]]

    def successes(self):
        return [row for row in self.records() if row["hard"] == 1]
