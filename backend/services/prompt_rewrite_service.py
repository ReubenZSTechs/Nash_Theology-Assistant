"""
Prompt rewriter — turns the user's latest chat message into a self-contained question.

Why: in a chat the user often writes follow-ups such as "and what does verse 29 add?".
A search engine (RAG) or a team of agents (Graph-of-Agents) cannot understand that message
without the earlier turns. The rewriter reads the recent history and produces e.g.

    history:  user: "What does Romans 3:28 teach?"   assistant: "...justification by faith..."
    message:  "and what does verse 29 add?"
    rewrite:  "What does Romans 3:29 add to Paul's teaching on justification by faith in Romans 3:28?"

This must run BEFORE the Graph-of-Agents tool (backend/configs/tools/run_GoA.py), which
only accepts a rewritten query.

The prompt and generation settings live in backend/configs/agent/main/prompt_rewriter.yaml.
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from backend.core.llm.agent_init import LLMAgent

logger = logging.getLogger(__name__)

REWRITER_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "agent" / "main" / "prompt_rewriter.yaml"
MAX_HISTORY_TURNS = 6          # only the most recent messages matter for a follow-up
MAX_TURN_CHARS = 800           # long assistant answers are cut so the prompt stays small


@lru_cache(maxsize=1)
def get_rewriter_agent() -> LLMAgent:
    return LLMAgent(agent_config=REWRITER_CONFIG)


def shorten(text: str, limit: int = MAX_TURN_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def format_history(history: list[dict]) -> str:
    """Write the last few turns as 'user: ...' / 'assistant: ...' lines."""
    recent = history[-MAX_HISTORY_TURNS:]
    return "\n".join(f"{turn.get('role', 'user')}: {shorten(turn.get('content', ''))}" for turn in recent)


def rewrite_prompt(message: str, history: Optional[list[dict]] = None) -> dict:
    """
    Rewrite `message` into a standalone question using the chat `history`.

    Args:
        message: the user's latest message.
        history: earlier turns, oldest first, as [{"role": "user" | "assistant", "content": "..."}].

    Returns {"original_message", "rewritten_query", "is_follow_up", "error"}.
    Without history there is nothing to resolve, so no LLM call is made. If the LLM call
    fails, the original message is used and `error` explains why (the pipeline keeps going).
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    message = message.strip()

    result = {"original_message": message, "rewritten_query": message, "is_follow_up": False, "error": None}
    if not history:
        return result

    user_prompt = f"Chat history:\n{format_history(history)}\n\nLatest message:\n{message}"
    try:
        output = get_rewriter_agent().generate_llm_response(user_prompt)
    except Exception as error:
        logger.exception("Prompt rewriting failed")
        result["error"] = f"prompt rewriting failed, using the original message: {error}"
        return result

    rewritten = (output.get("rewritten_query") or "").strip()
    result["rewritten_query"] = rewritten or message
    result["is_follow_up"] = bool(output.get("is_follow_up", False))
    return result
