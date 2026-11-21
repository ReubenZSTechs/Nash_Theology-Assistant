"""
Graph-of-Agents tool — lets the LLM ask a team of specialist theology agents to answer together.

ORDER MATTERS: this tool runs only AFTER the prompt_rewriter has turned the user's latest
message into a self-contained question (backend/services/prompt_rewrite_service.py).
That is why its only required argument is `rewritten_query`, and the description below
tells the LLM so. A follow-up such as "and verse 29?" must never reach this tool unrewritten.

How the tool call works (same as backend/configs/tools/retrieval.py):
  1. We send GOA_TOOL_SCHEMA in the `tools` list of a chat request.
  2. The LLM answers with a tool call:
         {"name": "run_graph_of_agents",
          "arguments": "{\"rewritten_query\": \"...\", \"k\": 3, \"pooling\": \"mean\"}"}
  3. Our code runs execute_goa_tool(arguments) and sends the result back as a "tool" message.

Over HTTP:
    POST /tools/goa          -> run the tool
    GET  /tools/goa/schema   -> get the tool definition
(The full trace of every agent is available at POST /goa/run.)
"""
import json
import logging
from typing import Literal, Union

from fastapi import APIRouter
from pydantic import BaseModel, Field, ValidationError

from backend.utils.goa_configs import DEFAULT_K, DEFAULT_POOLING, MAX_K, POOLING_METHODS

logger = logging.getLogger(__name__)

TOOL_NAME = "run_graph_of_agents"


# Schema for the LLM to choose
GOA_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Answer a theology or Bible question with Nash's Graph-of-Agents: a team of specialist "
            "Reformed agents (exegetes, systematic and biblical theologians, Westminster catechist, pastor, "
            "apologist, ethicist, historian, ...) that searches the knowledge base, answers, critiques and "
            "refines each other, and returns one combined answer with Scripture and Westminster references.\n"
            "Call this tool ONLY AFTER the prompt_rewriter has rewritten the user's latest message into a "
            "self-contained question using the chat history. Pass that rewritten question unchanged as "
            "`rewritten_query`; never pass a follow-up that depends on earlier messages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rewritten_query": {
                    "type": "string",
                    "description": "The self-contained question produced by the prompt_rewriter, e.g. "
                                   "'What does Romans 3:28 teach about justification by faith?'.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_K,
                    "default": DEFAULT_K,
                    "description": "How many specialist agents to involve. Use more for broad or difficult questions.",
                },
                "pooling": {
                    "type": "string",
                    "enum": list(POOLING_METHODS),
                    "default": DEFAULT_POOLING,
                    "description": "'mean' synthesises all agents' answers (better); 'max' returns the best agent's answer (faster).",
                },
            },
            "required": ["rewritten_query"],
        },
    },
}


# Input Validation
class GoAToolInput(BaseModel):
    rewritten_query: str = Field(..., min_length=1)
    k: int = Field(DEFAULT_K, ge=1, le=MAX_K)
    pooling: Literal["mean", "max"] = DEFAULT_POOLING


def parse_arguments(arguments: Union[str, dict, None]) -> GoAToolInput:
    """
    Turn the LLM's tool-call arguments into a validated object.

    The OpenAI API gives the arguments as a JSON *string*; HTTP callers send a dict.
    """
    if arguments is None or arguments == "":
        arguments = {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return GoAToolInput.model_validate(arguments)


def describe_validation_error(error: ValidationError) -> str:
    """Short, LLM-readable version of a Pydantic error, e.g. 'k: Input should be <= 5'."""
    messages = []
    for problem in error.errors():
        field = ".".join(str(part) for part in problem["loc"]) or "arguments"
        messages.append(f"{field}: {problem['msg']}")
    return "; ".join(messages)


# Executor for LLM when tools are called
def execute_goa_tool(arguments: Union[str, dict, None]) -> dict:
    """
    Run the Graph-of-Agents tool.

    Never raises: problems are returned as {"error": "..."} so the agent loop can pass
    the message back to the LLM. Only the parts the LLM needs are returned (not every
    intermediate answer), to keep the tool message small.
    """
    try:
        tool_input = parse_arguments(arguments)
    except json.JSONDecodeError as error:
        return {"error": f"arguments are not valid JSON: {error}"}
    except ValidationError as error:
        return {"error": f"invalid arguments: {describe_validation_error(error)}"}

    # Imported here, not at the top: the GoA pipeline itself imports the retrieval tool from
    # this package, so a top-level import would be circular.
    from backend.pipelines.agentic.GoA_pipeline import run_goa

    try:
        result = run_goa(tool_input.rewritten_query, k=tool_input.k, pooling=tool_input.pooling)
    except Exception as error:
        logger.exception("Graph-of-Agents tool failed")
        return {"error": f"graph-of-agents failed: {error}"}

    return {
        "query": result["query"],
        "final_answer": result["final_answer"],
        "selected_agents": result["selected_agents"],
        "ranking": result["ranking"],
        "relevance": result["relevance"],
        "sources": [{"label": source["label"], "title": source.get("reference") or source.get("title")}
                    for source in result["sources"]],
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
class GoAToolRequest(BaseModel):
    rewritten_query: str
    k: int = DEFAULT_K
    pooling: str = DEFAULT_POOLING


router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("/goa/schema")
def goa_schema_endpoint():
    return GOA_TOOL_SCHEMA


@router.post("/goa")
def goa_endpoint(request: GoAToolRequest):
    # Validation happens inside the executor, so HTTP callers get the same
    # {"error": ...} answers as the LLM does.
    return execute_goa_tool(request.model_dump())
