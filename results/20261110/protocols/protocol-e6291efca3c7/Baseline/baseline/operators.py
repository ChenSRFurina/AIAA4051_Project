"""Bounded generation using upstream prompts, JSON parsing and scope adoption."""
from graphskillevo.operators import EvolutionOperators, GraphSkillGenerationError, OperatorResult, extract_json
from graphskillevo.graph_skill import adopt_graph, adopt_other, assert_graph_skill
from graphskillevo.prompts import load_prompt


class CourseOperators(EvolutionOperators):
    def __init__(self, client, config):
        super().__init__(env_name=config["dataset"], max_completion_tokens=config["operator_max_tokens"],
                         reasoning_effort=None, graph_retry_attempts=1)
        self.client, self.config, self.call_seed = client, config, config["seed"]

    def _bounded_user(self, user, *, initializing=False):
        if not self.config["model"].get("structured_operators", False):
            return user
        initialization_format = (
            'For initialization, the JSON object must have exactly one key, "skills", '
            'whose value is an array of exactly one complete Markdown string. '
            'Do not add separate array entries for notes, punctuation, or JSON delimiters. '
            if self.config.get("backend", "local") == "api" else
            'For an initialization skills array, finish the Markdown string, then close the array and object: \"]}. '
        )
        if self.config.get("backend", "local") == "api" and not initializing:
            initialization_format = ""
        return user + (
            "\n\n## Serialization and size contract (common to all experiment methods)\n"
            f"The complete skill Markdown must fit within {self.config['skill_max_tokens']} tokens. "
            "Use at most six concise reusable nodes and at most three distinct task graphs. "
            "Do not enumerate many equivalent question categories or repeat the same workflow. "
            "Return exactly one JSON object, with no Markdown code fence or trailing prose. "
            + initialization_format
            + ("" if initializing and self.config.get("backend") == "api" else
               "For mutation, return the complete new_skill string and a short notes list. ") +
            "Every workflow item must be an exact defined node name; every defined node must be used. "
            "In Task Graphs, each task must start with its own '### <Task Name>' heading. "
            "The next line must be '**Use when:** <nonempty condition>', followed by '**Workflow:**' on its own line. "
            "Use when and Workflow are bold labels, never headings; do not prefix them with #. "
            "Workflow numbering must start at 1 and increase consecutively without duplicate numbers. "
            "After each **Workflow:** marker, include only numbered node names until the next task heading or end of document. "
            "Put all explanations, notes and fallback instructions in node definitions or Global Guidance, never after a workflow.\n"
            "Task graph syntax example (replace names and condition with your own, defined nodes):\n"
            "### Task Name\n"
            "**Use when:** The question matches this task.\n"
            "**Workflow:**\n"
            "1. Defined Node Name\n"
        )

    def validate(self, skill):
        assert_graph_skill(skill)
        if self.client.counter.count(skill) > self.config["skill_max_tokens"]:
            raise ValueError("Candidate exceeds the common skill token cap")

    def initial_attempt(self, base_skill, recent_errors=None):
        if self.env_name == "searchqa":
            task_context = load_prompt("task_context_searchqa", env="searchqa")
        else:
            task_context = "Answer multi-hop questions using supplied titled context passages. Return <answer> tags. Supporting facts and gold answers are scoring-only."
        user = self._build_init_user(n=1, base_skill=base_skill, task_context=task_context, recent_errors=recent_errors or [])
        user = self._bounded_user(user, initializing=True)
        system = load_prompt("init_population")
        if self.config.get("backend") == "api":
            system = system.replace('{"skills": ["full skill document", "..."]}',
                                    '{"skills": ["one complete Markdown skill document"]}')
            system += "\nFor this request, generate exactly one complete skill string. The skills array must contain exactly one string and no other elements."
        response, usage = self.client.chat(system, user, maximum=self.max_completion_tokens,
                                           stage="initialize", seed=self.call_seed, temperature=self.config["operator_temperature"])
        parsed = extract_json(response)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("skills"), list) or len(parsed["skills"]) != 1:
            raise GraphSkillGenerationError("Expected exactly one initial skill", prompt=user, raw_response=response, parsed=parsed)
        skill = str(parsed["skills"][0]).rstrip() + "\n"
        self.validate(skill)
        return skill, {"prompt": user, "raw_response": response, "usage": usage}

    def _single_skill_operator(self, *, system, user, stage, scope, parent_a):
        user = self._bounded_user(user)
        response, usage = self.client.chat(system, user, maximum=self.max_completion_tokens, stage=stage,
                                           seed=self.call_seed, temperature=self.config["operator_temperature"])
        parsed = extract_json(response)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("new_skill"), str) or not parsed["new_skill"].strip():
            raise GraphSkillGenerationError("Missing nonempty new_skill", prompt=user, raw_response=response, parsed=parsed)
        candidate = parsed["new_skill"].rstrip() + "\n"
        try:
            skill = adopt_other(parent_a, candidate) if scope == "other" else adopt_graph(parent_a, candidate)
            self.validate(skill)
        except ValueError as exc:
            raise GraphSkillGenerationError(str(exc), prompt=user, raw_response=response, parsed=parsed, usage=usage, candidate=candidate) from exc
        return OperatorResult(skill, user, response, parsed, usage)
