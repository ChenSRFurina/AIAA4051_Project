# Multi-Hop Question Answering Skill

## Global Guidance

### Overview
This skill enables agents to answer multi-hop questions in the HotpotQA benchmark by chaining reasoning steps over provided context passages. The agent acts as a logical reasoner that extracts facts, infers intermediate conclusions, and synthesizes final answers.

### General Principles
- **Context First**: Always rely strictly on the provided titled context passages. Do not use external knowledge unless explicitly permitted by the prompt.
- **Step-by-Step Inference**: Break complex questions into atomic steps. Solve sub-questions before combining results.
- **Evidence Verification**: Every claim must be supported by a specific sentence or fact in the context.
- **Exact Matching**: Final answers should match the text exactly unless formatting (like removing quotes) is necessary for clarity.
- **Error Recovery**: If a path leads to a contradiction or missing information, backtrack to the previous inference step and try an alternative path.

### Graph-structured Skill Execution Guide
This skill operates as a directed acyclic graph of reusable nodes.
- The **Global Guidance** section provides overarching rules for reasoning and context usage.
- The **Node Lists** section defines six reusable atomic steps: `Parse Goal`, `Retrieve Relevant Passages`, `Extract Facts`, `Infer Intermediate Conclusion`, `Synthesize Final Answer`, and `Verify Against Context`.
- The **Task Graphs** section maps specific question types to ordered workflows using these nodes.
- **Execution Protocol**: 
  1. Identify the question type using the `**Use when:**` condition in Task Graphs.
  2. Execute the workflow nodes in the numbered order specified.
  3. When executing a node (e.g., `1. Parse Goal`), follow the instructions under the `### Parse Goal` heading in Node Lists.
  4. Wrap the final result in `<answer>` tags.

## Node Lists

### Parse Goal
- Read the entire question carefully to identify the core entity and the relationship chain required.
- Determine if the question is a single-hop retrieval or a multi-hop chain requiring intermediate facts.
- Identify the target entity for the final answer.
- Note any constraints such as "according to the text" or specific entity types.

### Retrieve Relevant Passages
- Scan the provided titled context passages for keywords related to the entities and relationships identified in `Parse Goal`.
- Select passages that contain explicit mentions of the starting entity or the intermediate entities.
- Prioritize passages that link entities together (e.g., "X is a type of Y" or "X was born in Y").
- If no direct match exists, look for synonyms or cross-references within the titles and first sentences.

### Extract Facts
- From the selected passages, isolate specific sentences that state facts about the entities.
- Record the fact as a triplet: (Subject, Relation, Object) if the structure permits.
- Ensure the fact is explicitly stated in the text, not inferred.
- Discard irrelevant sentences that do not contribute to the logical chain.

### Infer Intermediate Conclusion
- Combine the extracted facts to form a logical bridge between the start and end of the query chain.
- If the question requires connecting Entity A to Entity B via Entity C, verify that the fact linking A to C and C to B both exist in the extracted facts.
- Formulate a temporary statement representing this intermediate connection.
- Check for consistency: ensure the intermediate entity matches in both facts.

### Synthesize Final Answer
- Based on the inferred intermediate conclusion, determine the final answer entity.
- Extract the exact string representing the answer from the context passages.
- If the answer is a list, ensure all items are included and formatted correctly.
- Prepare the answer string for output, ensuring it stands alone as a complete response.

### Verify Against Context
- Re-read the original question and the synthesized answer to ensure alignment.
- Trace the logical path one last time: Does the answer directly follow from the extracted facts?
- Check for hallucinations: Is every word in the answer present in the context?
- If verification fails, return to `Retrieve Relevant Passages` or `Extract Facts` to find missing links.

## Task Graphs

### Single-Hop Entity Retrieval
**Use when:** The question asks for a direct attribute, definition, or relationship of a single entity that is explicitly mentioned in the provided text without requiring intermediate steps.
**Workflow:**
1. Parse Goal
2. Retrieve Relevant Passages
3. Extract Facts
4. Synthesize Final Answer
5. Verify Against Context

### Multi-Hop Chain Reasoning
**Use when:** The question requires connecting multiple entities through a chain of relationships (e.g., "Who is the spouse of the director of Movie X?") where intermediate facts must be inferred.
**Workflow:**
1. Parse Goal
2. Retrieve Relevant Passages
3. Extract Facts
4. Infer Intermediate Conclusion
5. Synthesize Final Answer
6. Verify Against Context

### Complex Multi-Entity Comparison
**Use when:** The question requires comparing two or more entities or finding commonalities/differences between them based on the provided text.
**Workflow:**
1. Parse Goal
2. Retrieve Relevant Passages
3. Extract Facts
4. Infer Intermediate Conclusion
5. Synthesize Final Answer
6. Verify Against Context
