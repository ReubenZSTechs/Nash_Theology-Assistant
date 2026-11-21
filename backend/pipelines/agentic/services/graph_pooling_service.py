"""
Step 5 of Graph-of-Agents — graph pooling.

After message passing, every node of the graph holds a refined answer. Graph pooling
combines them into ONE final answer, like pooling node features in a graph neural network
(paper, Eq. 6):

  * max pooling   take the final answer of the most relevant agent (ranking[0]).
                  No extra LLM call, so it is fast.
  * mean pooling  the graph_pooler Meta-LLM reads all final answers with their relevance
                  scores and synthesises one answer that weighs them. It costs one extra LLM
                  call but uses every agent's insights. This is Nash's default.

Both methods return the same shape, so the caller does not care which one ran:
    {"method", "answer", "integrated_agents", "scripture_references",
     "confessional_references", "unresolved_disagreements", "confidence", "reasoning"}
"""
import logging

from backend.pipelines.agentic.services import prompts
from backend.pipelines.agentic.services.agent_pool import answer_text, format_response_for_peers, get_main_agent
from backend.utils.goa_configs import DEFAULT_POOLING, GRAPH_POOLER_AGENT, POOLING_METHODS

logger = logging.getLogger(__name__)


def max_pooling(responses: dict[str, dict], ranking: list[str]) -> dict:
    """Return the final answer of the most relevant agent, in the common output shape."""
    best_role = ranking[0]
    best = responses[best_role]
    return {
        "method": "max",
        "answer": answer_text(best),
        "integrated_agents": [best_role],
        "scripture_references": best.get("scripture_references", []),
        "confessional_references": best.get("confessional_references", []),
        "unresolved_disagreements": [],
        "confidence": best.get("confidence"),
        "reasoning": best.get("reasoning", ""),
    }


def format_weighted_responses(responses: dict[str, dict], relevance: dict[str, float], ranking: list[str]) -> str:
    """Write every final answer with its relevance score, most relevant first."""
    return "\n\n".join(
        f"[{role} — relevance score {relevance.get(role, 0.0):.2f}]\n{format_response_for_peers(responses[role])}"
        for role in ranking
    )


def mean_pooling(query: str, context: str, responses: dict[str, dict], relevance: dict[str, float], ranking: list[str]) -> dict:
    """Let the graph_pooler Meta-LLM synthesise all final answers into one."""
    output = get_main_agent(GRAPH_POOLER_AGENT).generate_llm_response(
        prompts.mean_pooling_prompt(query, context, format_weighted_responses(responses, relevance, ranking))
    )
    if not answer_text(output):
        raise ValueError("graph_pooler returned an empty answer")

    # The pooler may name an agent that is not in this graph; keep only real members.
    integrated = [role for role in output.get("integrated_agents", []) if role in ranking] or list(ranking)
    return {
        "method": "mean",
        "answer": answer_text(output),
        "integrated_agents": integrated,
        "scripture_references": output.get("scripture_references", []),
        "confessional_references": output.get("confessional_references", []),
        "unresolved_disagreements": output.get("unresolved_disagreements", []),
        "confidence": output.get("confidence"),
        "reasoning": output.get("reasoning", ""),
    }


def graph_pooling(query: str, responses: dict[str, dict], relevance: dict[str, float], ranking: list[str],
                  algorithm: str = DEFAULT_POOLING, context: str = "") -> tuple[dict, list[str]]:
    """
    Combine the agents' final answers into one answer.

    Returns (final_answer, errors). With a single agent there is nothing to combine, so max
    pooling is used. If mean pooling fails we fall back to max pooling and report it.
    """
    if algorithm not in POOLING_METHODS:
        raise ValueError(f"algorithm must be one of {POOLING_METHODS}, got {algorithm!r}")

    if algorithm == "max" or len(ranking) == 1:
        return max_pooling(responses, ranking), []

    try:
        return mean_pooling(query, context, responses, relevance, ranking), []
    except Exception as error:
        logger.exception("Mean pooling failed")
        return max_pooling(responses, ranking), [f"mean pooling failed, used max pooling instead: {error}"]
