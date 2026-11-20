"""
Retrieval tool — lets the LLM search Nash's knowledge base by itself.

How tool calling works (OpenAI-compatible API, e.g. vLLM):
  1. We send RETRIEVAL_TOOL_SCHEMA in the `tools` list of a chat request.
  2. When the LLM decides it needs information, it answers with a tool call:
         {"name": "search_knowledge_base",
          "arguments": "{\"query\": \"...\", \"vector_top_k\": 5, \"graph_top_k\": 3}"}
     The LLM chooses the query AND how many results it wants from each database.
  3. Our code runs execute_retrieval_tool(arguments) and sends the result back to the
     LLM as a "tool" message. The LLM then writes its answer using that context.

The same tool is also available over HTTP:
    POST /tools/retrieval          -> run the tool
    GET  /tools/retrieval/schema   -> get the tool definition
"""
import json
import logging
from typing import Union

from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError, model_validator

from backend.pipelines.rag.rag_pipeline import retrieve
from backend.utils.rag_configs import DEFAULT_TOP_K, MAX_TOP_K

logger = logging.getLogger(__name__)

TOOL_NAME = "search_knowledge_base"


# Schema tools for the LLM to choose
RETRIEVAL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Search Nash's Reformed theology knowledge base. It combines two searches:\n"
            "- vector database: sermon transcript passages and the Westminster Standards "
            "(Confession, Larger and Shorter Catechism). Best for doctrine, explanations and preaching.\n"
            "- knowledge graph: Bible verses (BSB and Indonesian TB) plus their cross-references, "
            "and which sermons / Westminster sections cite them. Best for scripture and related verses.\n"
            "Choose how many results to take from each. Use 0 to skip one of them. "
            "Write Bible references in the query (e.g. 'Romans 3:28') to start the graph search from those verses. "
            "Results are labelled [V1].. (vector) and [G1].. (graph); cite these labels in the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A clear, self-contained search query, e.g. 'What is justification by faith? Romans 3:28'.",
                },
                "vector_top_k": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_TOP_K,
                    "default": DEFAULT_TOP_K,
                    "description": "Number of sermon / Westminster passages to return from the vector database (0 to skip).",
                },
                "graph_top_k": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_TOP_K,
                    "default": DEFAULT_TOP_K,
                    "description": "Number of Bible verses to return from the knowledge graph (0 to skip).",
                },
            },
            "required": ["query"],
        },
    },
}


# Input Validation
class RetrievalToolInput(BaseModel):
    query: str = Field(..., min_length=1)
    vector_top_k: int = Field(DEFAULT_TOP_K, ge=0, le=MAX_TOP_K)
    graph_top_k: int = Field(DEFAULT_TOP_K, ge=0, le=MAX_TOP_K)

    @model_validator(mode="after")
    def at_least_one_branch(self):
        if self.vector_top_k == 0 and self.graph_top_k == 0:
            raise ValueError("at least one of vector_top_k and graph_top_k must be greater than 0")
        return self


def parse_arguments(arguments: Union[str, dict, None]) -> RetrievalToolInput:
    """
    Turn the LLM's tool-call arguments into a validated object.

    The OpenAI API gives the arguments as a JSON *string*; HTTP callers send a dict.
    """
    if arguments is None or arguments == "":
        arguments = {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return RetrievalToolInput.model_validate(arguments)


def describe_validation_error(error: ValidationError) -> str:
    """Short, LLM-readable version of a Pydantic error, e.g. 'vector_top_k: Input should be <= 20'."""
    messages = []
    for problem in error.errors():
        field = ".".join(str(part) for part in problem["loc"]) or "arguments"
        messages.append(f"{field}: {problem['msg']}")
    return "; ".join(messages)


# Executor for LLM when tools are called
def execute_retrieval_tool(arguments: Union[str, dict, None]) -> dict:
    """
    Run the retrieval tool.

    Never raises: problems are returned as {"error": "..."} so the agent loop can pass
    the message back to the LLM, which may then retry with better arguments.
    """
    try:
        tool_input = parse_arguments(arguments)
    except json.JSONDecodeError as error:
        return {"error": f"arguments are not valid JSON: {error}"}
    except ValidationError as error:
        return {"error": f"invalid arguments: {describe_validation_error(error)}"}

    try:
        result = retrieve(tool_input.query, tool_input.vector_top_k, tool_input.graph_top_k)
    except Exception as error:
        logger.exception("Retrieval tool failed")
        return {"error": f"retrieval failed: {error}"}

    return {
        "query": result["query"],
        "context": result["context"],
        "sources": result["sources"],
        "errors": result["errors"],
    }


def to_tool_message(tool_call_id: str, result: dict) -> dict:
    """Wrap a tool result as the chat message that is sent back to the LLM."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(result, ensure_ascii=False, default=str),
    }


# Endpoint
class RetrievalToolRequest(BaseModel):
    query: str
    vector_top_k: int = DEFAULT_TOP_K
    graph_top_k: int = DEFAULT_TOP_K


router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/retrieval/schema")
def retrieval_schema_endpoint():
    return RETRIEVAL_TOOL_SCHEMA


@router.post("/retrieval")
def retrieval_endpoint(request: RetrievalToolRequest):
    # Validation happens inside the executor, so HTTP callers get the same
    # {"error": ...} answers as the LLM does.
    return execute_retrieval_tool(request.model_dump())
