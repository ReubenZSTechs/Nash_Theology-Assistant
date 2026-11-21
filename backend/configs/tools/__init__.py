"""
Tools the LLM can call.

TOOL_SCHEMAS is sent to the LLM in the `tools` list of a chat request.
TOOL_EXECUTORS maps a tool name to the Python function that runs it.

Keep the retrieval import first: the Graph-of-Agents pipeline imports the retrieval tool,
and run_GoA only imports that pipeline lazily (inside its executor).
"""
from backend.configs.tools.retrieval import RETRIEVAL_TOOL_SCHEMA, TOOL_NAME as RETRIEVAL_TOOL_NAME, execute_retrieval_tool
from backend.configs.tools.run_GoA import GOA_TOOL_SCHEMA, TOOL_NAME as GOA_TOOL_NAME, execute_goa_tool

TOOL_SCHEMAS = [RETRIEVAL_TOOL_SCHEMA, GOA_TOOL_SCHEMA]

TOOL_EXECUTORS = {
    RETRIEVAL_TOOL_NAME: execute_retrieval_tool,
    GOA_TOOL_NAME: execute_goa_tool,
}
