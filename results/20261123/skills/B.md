# SearchQA Evidence Retrieval Skill

## Global Guidance

### Overview
This skill enables an agent to answer questions by strictly adhering to provided context passages. The agent must locate specific evidence within the text and formulate a concise answer based solely on that information, avoiding hallucination or external knowledge.

### General Principles
- **Strict Adherence**: Answers must be derived exclusively from the provided context. Do not use external knowledge or assume facts not present.
- **Entity Extraction**: When the question asks for a specific entity (e.g., a name, place, or person), extract only the entity name itself from the supporting evidence, not the entire sentence or descriptive phrases surrounding it.
- **Conciseness**: The final answer inside `<answer>` tags should be a short phrase or sentence, avoiding repetition of the question or excessive explanation.
- **Exact Matching**: Ensure the extracted answer matches the gold standard's expected entity or fact precisely. Do not include titles, descriptions, or context unless the gold answer explicitly includes them.
- **Fallback Strategy**: If the context does not contain the answer, explicitly state that the information is not available in the provided text.

### Graph-structured Skill Execution Guide
This skill operates by selecting a relevant workflow from `## Task Graphs` based on the question type, then executing the corresponding sequence of nodes defined in `## Node Lists`.

1. Analyze the QUESTION to determine if it requires simple retrieval or multi-step reasoning.
2. Select the matching Task Graph workflow (e.g., `Direct Fact Retrieval` or `Multi-Step Reasoning`).
3. Execute the numbered nodes in the selected workflow sequentially.
4. Apply the instructions within each node to perform the specific action (e.g., scanning, extracting, validating).
5. Format the final result using the `<answer>` tags as required.
6. Ensure the content inside `<answer>` tags is a direct extraction or concise summary of the evidence, not a restatement of the question.

## Node Lists

### Analyze Question Intent
- Read the QUESTION carefully to identify the core information needed.
- Determine if the answer is a single explicit fact, a combination of facts, or requires a comparison/calculation.
- Identify keywords in the question that should be used to scan the CONTEXT.

### Scan Context for Keywords
- Read through the provided CONTEXT passages systematically.
- Look for sentences containing the keywords identified in the `Analyze Question Intent` node.
- Highlight or mentally note any candidate sentences that appear to answer the question.

### Extract Supporting Evidence
- Select the most relevant sentence or phrase from the CONTEXT that directly answers the question.
- Ensure the evidence is complete and self-contained.
- If multiple sentences are needed, extract them all to form the evidence block.

### Validate Evidence Against Question
- Compare the extracted evidence with the original QUESTION to ensure it addresses all parts of the query.
- Verify that the evidence does not contradict the question or contain irrelevant information.
- If the evidence is insufficient or missing, note that the answer cannot be derived from the text.

### Formulate Concise Answer
- Synthesize the validated evidence into a direct, concise answer.
- Remove any introductory phrases or context that is not part of the direct answer.
- Ensure the answer is a few words or a short phrase, not a full paragraph.

### Format Final Response
- Enclose the concise answer within `<answer>...</answer>` tags.
- Ensure no markdown formatting (like bolding or lists) is used inside the tags unless explicitly part of the text.
- Double-check that the content inside the tags does not repeat the question.

## Task Graphs

### Direct Fact Retrieval
**Use when:** The question asks for a single specific fact, definition, or value explicitly stated in the context.
**Workflow:**
1. Analyze Question Intent
2. Scan Context for Keywords
3. Extract Supporting Evidence
4. Validate Evidence Against Question
5. Formulate Concise Answer
6. Format Final Response

### Multi-Step Reasoning
**Use when:** The question requires combining multiple pieces of information, performing a comparison, or deriving an answer through logical steps within the context.
**Workflow:**
1. Analyze Question Intent
2. Scan Context for Keywords
3. Extract Supporting Evidence
4. Validate Evidence Against Question
5. Formulate Concise Answer
6. Format Final Response

### Information Not Found
**Use when:** The agent has scanned the entire context and cannot locate evidence to answer the question.
**Workflow:**
1. Analyze Question Intent
2. Scan Context for Keywords
3. Extract Supporting Evidence
4. Validate Evidence Against Question
5. Formulate Concise Answer
6. Format Final Response
