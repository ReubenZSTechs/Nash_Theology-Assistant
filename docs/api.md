# Nash API

Start the server:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Every route is under `/api`, because nginx forwards `/api/...` to the backend unchanged.
Interactive docs are at `http://localhost:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | the API process is up |
| POST | `/api/rag/retrieve` | full hybrid RAG (vector database + knowledge graph) |
| POST | `/api/rag/vector-search` | Advanced RAG only (sermons + Westminster) |
| POST | `/api/rag/graph-search` | Graph RAG only (Bible verses + cross-references) |
| POST | `/api/rag/rerank` | rerank any list of texts with the LLM |
| GET | `/api/rag/health` | checks the embedding server, the LLM (reranker), Neo4j and both Postgres databases |
| POST | `/api/tools/retrieval` | runs the LLM retrieval tool |
| GET | `/api/tools/retrieval/schema` | the tool definition to send to the LLM |

How the pipeline works is explained in [pipeline.md](pipeline.md).

---

## RAG endpoints (`backend/pipelines/rag/rag_pipeline.py`)

### `POST /api/rag/retrieve`

| Field | Type | Default | Rule |
|---|---|---|---|
| `query` | string | required | not empty |
| `vector_top_k` | int | 5 | 0–20, 0 skips the vector database |
| `graph_top_k` | int | 5 | 0–20, 0 skips the knowledge graph |
| `rerank` | bool | `true` (env `RAG_RERANK`) | let the LLM re-score the vector candidates |

At least one of the two `top_k` values must be greater than 0.

```bash
curl -X POST http://localhost:8000/api/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What is justification by faith? Romans 3:28", "vector_top_k": 3, "graph_top_k": 4}'
```

Response (shortened):

```json
{
  "query": "What is justification by faith? Romans 3:28",
  "context": "[V1] Westminster Standards — Westminster Shorter Catechism WSC.33\nQ. What is justification? ...\n\n[G1] Bible — Romans 3:28 ...",
  "sources": [
    {"label": "V1", "id": "WSC.33", "source": "westminster", "title": "...", "text": "...", "url": null,
     "similarity": 0.60, "rrf_score": 0.0325, "rerank_score": 0.94},
    {"label": "G1", "id": "Rom.3.28", "source": "bible", "reference": "Romans 3:28", "pericope": "...",
     "text": "...", "text_tb": "...", "score": 1.0, "relation": "referenced in the query",
     "linked_sources": [{"source": "westminster", "id": "WCF.11.1", "title": "..."}]}
  ],
  "vector_results": ["... same items as the V-labelled sources, without labels ..."],
  "graph_results": ["... same items as the G-labelled sources, without labels ..."],
  "errors": []
}
```

The response fields are:

- **`context`**: plain text ready to put into an LLM prompt.
- **`sources`**: the same items as structured data. Items with a `V` label come from the vector database, and
  items with a `G` label come from the knowledge graph.
- **`errors`**: non-fatal problems. For example, if Neo4j is down, the vector results are still returned and
  the reason is listed here.

Status codes:

- **422**: invalid input.
- **503**: the pipeline could not run at all, usually because the embedding server is unreachable.

### `POST /api/rag/vector-search` and `POST /api/rag/graph-search`

Both take `{"query": "...", "top_k": 5}` (`top_k` from 1 to 20) and return the same response shape as
`/retrieve`, with only one branch filled. `/vector-search` also accepts `"rerank": true | false`.

### `POST /api/rag/rerank`

This endpoint reranks **any** list of texts with the LLM, using the same reranker as the pipeline (see
[pipeline.md](pipeline.md#llm-reranking-servicesrerank_servicepy)). The request and response follow the
common rerank API shape (`index`, `relevance_score`).

| Field | Type | Default | Rule |
|---|---|---|---|
| `query` | string | required | not empty |
| `documents` | list of strings | required | 1–50 non-empty texts |
| `top_n` | int | all | return only the best N |

```bash
curl -X POST http://localhost:8000/api/rag/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "What is justification by faith?",
       "documents": ["The Lord'"'"'s Supper is a sacrament...",
                     "Q. What is justification? A. Justification is an act of God'"'"'s free grace...",
                     "sebab itu kita yang dibenarkan karena iman kita hidup dalam damai sejahtera dengan Allah"]}'
```

```json
{"model": "llm",
 "results": [
   {"index": 1, "relevance_score": 0.8161, "document": "Q. What is justification? ..."},
   {"index": 2, "relevance_score": 0.472,  "document": "sebab itu kita yang dibenarkan ..."},
   {"index": 0, "relevance_score": 0.002,  "document": "The Lord's Supper is a sacrament..."}]}
```

`index` is the position of the document in the request, and `relevance_score` runs from 0 to 1, best first.
Status codes:
- **422**: invalid input.
- **503**: the LLM server is unreachable.

### `GET /api/rag/health`

```json
{"status": "degraded",
 "components": {"embedding": "ok", "reranker_llm": "ok", "neo4j": "error: Couldn't connect to localhost:7687 ...",
                "sermon_db": "ok", "westminster_db": "ok"}}
```

### Using the services from Python

```python
from backend.pipelines.rag.rag_pipeline import retrieve, vector_search, graph_search, rerank, health

result = retrieve("What is the chief end of man?", vector_top_k=5, graph_top_k=3)   # reranks by default
print(result["context"])

ranking = rerank("What is the chief end of man?", ["To glorify God...", "Bread and wine..."], top_n=1)
```

---

## Retrieval tool (`backend/configs/tools/retrieval.py`)

The tool lets the LLM decide **when** to search, **what** to search for, and **how many** results it wants
from each database.

### Tool definition (`GET /api/tools/retrieval/schema`)

```json
{
  "type": "function",
  "function": {
    "name": "search_knowledge_base",
    "description": "Search Nash's Reformed theology knowledge base ...",
    "parameters": {
      "type": "object",
      "properties": {
        "query":        {"type": "string"},
        "vector_top_k": {"type": "integer", "minimum": 0, "maximum": 20, "default": 5},
        "graph_top_k":  {"type": "integer", "minimum": 0, "maximum": 20, "default": 5}
      },
      "required": ["query"]
    }
  }
}
```

### `POST /api/tools/retrieval`

```bash
curl -X POST http://localhost:8000/api/tools/retrieval \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the Holy Spirit?", "vector_top_k": 5, "graph_top_k": 0}'
```

On success it returns `{"query", "context", "sources", "errors"}`. The tool **never raises**: bad arguments or
a failed search return `{"error": "..."}`, so the LLM can read the problem and try again. For example:

```json
{"error": "invalid arguments: vector_top_k: Input should be less than or equal to 20"}
```

### Using the tool in an agent loop

```python
import json
from openai import OpenAI

from backend.configs.tools import TOOL_EXECUTORS, TOOL_SCHEMAS
from backend.configs.tools.retrieval import to_tool_message

client = OpenAI(base_url="http://localhost:15000/v1", api_key="not-needed")
messages = [{"role": "user", "content": "What does the Bible teach about justification? Cite your sources."}]

while True:
    response = client.chat.completions.create(model="llm", messages=messages, tools=TOOL_SCHEMAS)
    message = response.choices[0].message
    if not message.tool_calls:
        print(message.content)          # final answer, citing [V1], [G2], ...
        break

    messages.append(message)
    for tool_call in message.tool_calls:
        execute = TOOL_EXECUTORS[tool_call.function.name]
        result = execute(tool_call.function.arguments)   # the raw JSON string from the LLM
        messages.append(to_tool_message(tool_call.id, result))
```

### How vLLM handles tool calls

The `llm` service in `docker-compose.yaml` is started with:

```
--enable-auto-tool-choice      # allow tool_choice="auto" (the model decides)
--tool-call-parser llama3_json # how to read a tool call out of Llama 3.1's text output
--chat-template /vllm-workspace/examples/tool_chat_template_llama3.1_json.jinja
```

`--chat-template` is needed because the NVFP4 checkpoint ships a minimal template that ignores `tools`.
Without it, the model would never see the tool definitions.

**vLLM does not choose the tool; the model does.** vLLM only translates between the API and the model:

1. **Prompt.** The request's `tools` list is rendered into the prompt by the chat template. For Llama 3.1 it
   goes into the first user turn, as the JSON schema plus the instruction *"Respond in the format
   {"name": function name, "parameters": {...}}"*.
2. **Decision.** The model reads the tool `description`, the parameter descriptions and the question. It then
   either answers in plain text, or writes a JSON tool call such as
   `{"name": "search_knowledge_base", "parameters": {"query": "...", "vector_top_k": 3, "graph_top_k": 6}}`.
   The values of `vector_top_k` and `graph_top_k` are chosen by the model here, guided by the descriptions in
   `RETRIEVAL_TOOL_SCHEMA`. Better descriptions lead to better choices.
3. **Parsing.** The `llama3_json` parser detects that JSON and returns it as
   `message.tool_calls[i].function.name` and `.arguments` (a JSON string), with `finish_reason="tool_calls"`.
4. **Execution (our code).** vLLM never runs anything. Our agent loop looks the name up in `TOOL_EXECUTORS`,
   runs it, and sends the result back as a `role: "tool"` message. The model then writes the final answer.

The `tool_choice` field of the request controls step 2:

| `tool_choice` | Behaviour |
|---|---|
| `"auto"` (default when `tools` is given) | the model decides; needs `--enable-auto-tool-choice` + parser |
| `"required"` | the model must call some tool; vLLM enforces valid JSON with guided decoding |
| `{"type": "function", "function": {"name": "search_knowledge_base"}}` | forces this tool; arguments are guaranteed to match the schema |
| `"none"` | no tool call |

Notes for Llama 3.1 8B:
- In `auto` mode it sometimes answers from memory instead of searching. If retrieval must always happen,
  force the tool on the first turn, then use `auto` afterwards.
- In `auto` mode nothing stops the model from writing invalid arguments, such as `vector_top_k: 50`. The
  executor's Pydantic validation is the real safety net: it returns `{"error": ...}` and the model can retry.
- `LLMAgent` in `backend/core/llm/agent_init.py` always sends `guided_json` / `response_format`, which
  forces JSON output and blocks tool calls. Use a plain `client.chat.completions.create(...)` call (as above)
  for the tool loop.

### Registering a new tool

vLLM has no server-side tool registry: tools are sent **with every request**. Registration only happens in
our code, in `backend/configs/tools/__init__.py`:

```python
TOOL_SCHEMAS = [RETRIEVAL_TOOL_SCHEMA]                         # what the LLM sees
TOOL_EXECUTORS = {RETRIEVAL_TOOL_NAME: execute_retrieval_tool}  # what our code runs
```

To add a tool:

1. Create `backend/configs/tools/<tool_name>.py` with:
   - a `TOOL_NAME`;
   - a schema dict in the OpenAI format (`{"type": "function", "function": {"name", "description", "parameters"}}`);
   - an executor `def execute_<tool_name>(arguments: str | dict) -> dict` that validates its input and returns
     `{"error": ...}` instead of raising.
2. Import them in `backend/configs/tools/__init__.py`, append the schema to `TOOL_SCHEMAS` and add
   `TOOL_NAME: executor` to `TOOL_EXECUTORS`.
3. Optionally expose it over HTTP with an `APIRouter` and include it in `backend/api/main.py`, as
   `retrieval.py` does.

The agent loop above needs no change: it sends `TOOL_SCHEMAS` and dispatches with `TOOL_EXECUTORS`.
