"""
Step 2 of Graph-of-Agents — initial responses.

Every selected agent answers the question on its own, in parallel, using the same
retrieved context. Each agent answers in its own JSON schema (configs/schema/hub/<role>.schema.json),
and every schema contains at least "reasoning", "answer" and "confidence".

These first answers are what the agents will score in edge sampling.
"""
import logging

from backend.pipelines.agentic.services import prompts
from backend.pipelines.agentic.services.agent_pool import get_hub_agent, run_in_parallel

logger = logging.getLogger(__name__)


def ask_agent(role: str, user_prompt: str) -> dict:
    """Send one prompt to one hub agent and return its parsed JSON answer."""
    return get_hub_agent(role).generate_llm_response(user_prompt)


def generate_initial_responses(roles: list[str], query: str, context: str) -> tuple[dict[str, dict], list[str]]:
    """
    Ask every agent in `roles` for its first answer, all at the same time.

    Returns (responses, errors):
        responses = {role: answer JSON} for the agents that answered successfully
        errors    = one message per agent that failed (that agent is left out of the graph)
    """
    user_prompt = prompts.initial_response_prompt(query, context)
    results = run_in_parallel(lambda role: ask_agent(role, user_prompt), roles)

    responses, errors = {}, []
    for role, result in results.items():
        if isinstance(result, Exception):
            logger.warning("Agent %s failed to answer: %s", role, result)
            errors.append(f"agent '{role}' failed to give an initial response and was removed: {result}")
        else:
            responses[role] = result

    return responses, errors
