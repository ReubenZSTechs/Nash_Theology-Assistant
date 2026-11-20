"""
Tools the LLM can call.

TOOL_SCHEMAS is sent to the LLM in the `tools` list of a chat request.
TOOL_EXECUTORS maps a tool name to the Python function that runs it.
"""
from backend.configs.tools.retrieval import RETRIEVAL_TOOL_SCHEMA, TOOL_NAME as RETRIEVAL_TOOL_NAME, execute_retrieval_tool

TOOL_SCHEMAS = [RETRIEVAL_TOOL_SCHEMA]

TOOL_EXECUTORS = {
    RETRIEVAL_TOOL_NAME: execute_retrieval_tool,
}
