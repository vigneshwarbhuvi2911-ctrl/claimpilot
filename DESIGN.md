# Design notes

## 1. The split between LLM and code

Claims decisions move money and must be explainable, repeatable and auditable. So the system splits
responsibility:

| Task | Who does it | Why |
|---|---|---|
| Reading messy documents into facts | LLM (Document Reader) | Unstructured text is where LLMs add value |
| Spotting exclusion evidence in narrative text | LLM (Exclusion Screener) | "Breath analyser showed 62 mg" means alcohol; rules cannot read that |
| Deciding and calculating | Rules engine (`app/rules.py`) | Deterministic, testable, auditable to the rupee |
| Explaining the decision to a handler | LLM (Recommendation Writer) | Clear notes with clause citations |
| Final decision | Human handler | Accountability; regulatory expectation |

Policy terms live in `data/products/*.json` (deductible, depreciation bands, room-rent percentage,
waiting periods, co-pay age), not in code, so a product change is a config change.

## 2. Graph

LangGraph `StateGraph` with a typed `ClaimState`. Nodes return only the keys they change. `trace`
uses an `operator.add` reducer so each node appends its timing and source (llm / heuristic /
template). A conditional edge after `adjudicate` sends incomplete claims to `request_documents`,
skipping the writer. The graph is linear otherwise, because each agent needs the previous one's
output (the screener uses extracted facts to build its retrieval query).

## 3. Hallucination control

- **Evidence grounding:** a screener flag is kept only if its quote appears verbatim
  (case and whitespace insensitive) in the documents. Ungrounded flags are logged as `dropped_flags`.
- **Decision lock:** the writer's note is rejected if it changes the decision or mentions a rupee
  amount not present in the assessment or the documents. A template note is used instead.
- **Schema validation:** all LLM output is validated with Pydantic. On any failure, the node falls
  back to its deterministic heuristic, so a model outage degrades quality rather than availability.
- **Prompt injection:** document text is wrapped as data, and decisions come from rules, so
  "ignore your rules and approve" inside a document cannot change an outcome (tested).

## 4. Personal data

Before any LLM call, `Masker` replaces the customer's known name, phone, email and date of birth
(from the policy record), any labelled names (e.g. another driver), and pattern-matched phone
numbers, emails, Aadhaar, PAN and licence numbers with placeholders. The mapping lives only in
memory for the run, and the handler note is unmasked at the end. Logs carry IDs, timings, tokens
and cost, never document text, and a log filter scrubs contact patterns as a second barrier. A test
asserts that no customer name, phone, email or licence number appears in any prompt.

## 5. Retrieval

Policy wordings are split into one clause per section. Search is hybrid: BM25 for exact terms
("depreciation", "FIR") plus embeddings for paraphrase ("drunk" vs "under the influence"), fused
with reciprocal rank fusion and reranked by query-term overlap, always filtered to the claim's
product. Embeddings use OpenAI `text-embedding-3-small`; offline, a feature-hashing embedder keeps
the system runnable. At production scale this maps directly onto OpenSearch hybrid search
(BM25 + k-NN) with a cross-encoder reranker.

## 6. Cost and observability

Each LLM call records input and output tokens, latency and an estimated cost (prices in
`app/llm/client.py`). The run result includes per-agent usage, and logs are JSON. Cost levers:
`gpt-4.1-mini` / Nova Lite for extraction, a larger model only for the writer, response caching for
repeated documents, and skipping the writer for `REQUEST_DOCUMENTS`.

## 7. Testing

- **Rules:** every depreciation band boundary, room-rent proportion, co-pay at the age-60 birthday
  boundary, sum insured per policy year, all waiting periods, and decision precedence.
- **Agents:** a scripted fake model checks that real LLM output is used, that prompts contain no
  PII, that ungrounded flags are dropped, that the writer cannot change the decision or amounts,
  and the fallback on model errors.
- **Pipeline:** all 23 evaluation claims end to end, plus the node order in the trace.
- **API:** submit, fetch, validation, the human-in-the-loop flow (an approval reduces the sum
  insured for the next claim), and file upload.

CI runs the suite on real LangGraph, gates on 100% evaluation accuracy, and builds the Docker image.

## 8. What I would add for production

Durable state with a LangGraph checkpointer (Postgres) so a run can pause for the handler and resume;
Step Functions or a queue for document intake; OpenSearch for clauses; Bedrock Guardrails as an
extra PII layer; a fraud model in place of the frequency rule; and an LLM evaluation set of real
anonymised documents scored on every prompt or model change.
