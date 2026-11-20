"""Export actual ordered task graphs; preserve conditions without inventing branches."""
import re
from pathlib import Path
from graphskillevo.graph_skill import assert_graph_skill, split_graph_skill
from .io import digest, write_json


def blocks(section):
    matches = list(re.finditer(r"(?m)^###\s+(.+)$", section))
    return [(m.group(1).strip(), section[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(section)].strip())
            for i, m in enumerate(matches)]


def label(value):
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")


def export_skill(skill, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".md").write_text(skill, encoding="utf-8")
    if not skill.strip():
        output.with_suffix(".mmd").write_text('flowchart TD\n  QA["No skill: base QA instructions only"]\n', encoding="utf-8")
        write_json(output.with_suffix(".json"), {"hash": digest(skill), "nodes": {}, "routes": [], "condition": "no_skill"})
        return
    assert_graph_skill(skill)
    parts = split_graph_skill(skill)
    nodes = dict(blocks(parts.node_lists))
    routes = []
    lines = ["flowchart TD", '  START["Task-type routing"]']
    for index, (name, body) in enumerate(blocks(parts.task_graphs)):
        condition = re.search(r"\*\*Use when:\*\*\s*(.+)", body).group(1).strip()
        sequence = re.findall(r"(?m)^\d+\.\s+(.+)$", body)
        routes.append({"task_type": name, "use_when": condition, "sequence": sequence})
        previous = f"r{index}"
        lines.extend([f'  {previous}["{label(name)}: {label(condition)}"]', f"  START --> {previous}"])
        for pos, node in enumerate(sequence):
            current = f"r{index}n{pos}"
            lines.extend([f'  {current}["{label(node)}"]', f"  {previous} --> {current}"])
            previous = current
    if not routes:
        lines.append('  START --> EMPTY["Empty upstream initial template"]')
    output.with_suffix(".mmd").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output.with_suffix(".json"), {"hash": digest(skill), "nodes": nodes, "routes": routes,
               "branch_policy": "Ordered workflows and routing are literal. Any conditions inside node instructions remain in node text; no internal control flow is inferred."})


def export_lineage(history, output):
    lines = ["flowchart LR"]
    for record in history:
        keep = set(record["survivor_ids"])
        for item in record["candidates"]:
            node = item["id"]
            score = item.get("val_hard")
            lines.append(f'  {node}["{node} EM={score} retained={node in keep}"]')
            for parent in item["parent_ids"]:
                lines.append(f"  {parent} --> {node}")
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
