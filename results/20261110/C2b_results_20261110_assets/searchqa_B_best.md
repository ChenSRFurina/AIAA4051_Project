# SearchQA Question Answering Skill

## Global Guidance

### Overview
This skill supports agents in the SearchQA benchmark. The agent receives a CONTEXT of document passages and a QUESTION. The goal is to locate evidence within the context and formulate a concise answer based strictly on that evidence.

### General Principles
- **Context First**: Always prioritize the provided CONTEXT over external knowledge.
- **Evidence-Based**: Answers must be derived solely from the supplied passages. If information is absent, explicitly state "Not found in context" inside the answer tags.
- **Conciseness**: Final answers should be short phrases or sentences, enclosed in `<answer>...</answer>` tags. Do not add extra explanation outside the tags.
- **Strict Adherence**: Do not hallucinate answers or use outside knowledge even if the context seems incomplete or ambiguous.
- **Exact Matching**: The final answer must match the gold standard exactly. Do not output the question text, the entire context passage, or boolean values (True/False) unless the gold answer is exactly that. Extract the specific entity or fact name requested.

### Graph-structured Skill Execution Guide
This skill uses a graph of reusable nodes to solve tasks.
- **Global Guidance** applies to all operations.
- **Node Lists** define reusable steps like parsing, searching, extracting, verifying, and formulating.
- **Task Graphs** define workflows by selecting the appropriate task type and executing the ordered list of node names.
- To execute: Identify the relevant **Task Graph**, then follow its **Workflow** (e.g., `1. Parse Question`, `2. Search Context`) using the instructions in **Node Lists**.

## Node Lists

### Parse Question
- Extract the core intent and key entities from the QUESTION.
- Identify the specific information required to form an answer.
- Note any constraints on the answer format.

### Search Context
- Scan the provided CONTEXT passages for keywords or concepts matching the parsed question intent.
- Retrieve relevant sentences or paragraphs that appear to contain the answer.
- If multiple passages are relevant, prioritize the one with the most direct information.

### Extract Evidence
- Isolate the specific sentence or phrase within the retrieved context that directly answers the question.
- Ignore surrounding noise or unrelated text in the passage.
- Ensure the extracted text is a complete thought or fact.

### Verify Evidence
- Cross-check the extracted evidence against the original question to ensure it fully addresses the intent.
- Confirm that the evidence is explicitly present in the provided CONTEXT.
- If the evidence is ambiguous or partial, flag it for the next step.

### Formulate Answer
- Synthesize the verified evidence into a concise, direct answer.
- Remove any context-specific references (e.g., "According to the text...") unless necessary for clarity.
- Ensure the answer is a few words or a short phrase.

### Wrap Answer
- Enclose the final concise answer in `<answer>...</answer>` tags.
- If no evidence was found, enclose "Not found in context" in the tags.
- Do not add extra explanation outside the tags.

## Task Graphs

### Direct Evidence Question
**Use when:** The question can be answered by locating a specific fact or phrase directly within the context.
**Workflow:**
1. Parse Question
2. Search Context
3. Extract Evidence
4. Verify Evidence
5. Formulate Answer
6. Wrap Answer

### Multi-Hop Fact Retrieval
**Use when:** The question requires combining information from different parts of the context or inferring a relationship between two entities mentioned in the text.
**Workflow:**
1. Parse Question
2. Search Context
3. Extract Evidence
4. Verify Evidence
5. Formulate Answer
6. Wrap Answer

### Negative Evidence Query
**Use when:** The question asks for information that might not be present in the context, or asks for a negation (e.g., "What is NOT included...").
**Workflow:**
1. Parse Question
2. Search Context
3. Verify Evidence
4. Formulate Answer
5. Wrap Answer

### Implicit Relationship Query
**Use when:** The question asks about a relationship or calculation not explicitly stated as a sentence but derivable from the context (e.g., comparing values).
**Workflow:**
1. Parse Question
2. Search Context
3. Extract Evidence
4. Verify Evidence
5. Formulate Answer
6. Wrap Answer
