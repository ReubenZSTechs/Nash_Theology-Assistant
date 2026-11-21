"""
Step 1 of Graph-of-Agents — node sampling.

Calling all 40 agents for every question would be slow and noisy ("agent explosion").
Instead a Meta-LLM (the node_sampler agent) reads the question and the agents' model
cards and picks only the k most suitable agents:

    V_s = MetaLLM(Top-k | question, model cards)

Example: "How should I comfort a grieving friend?" -> ["pastor", "general_theologian", "devotional_writer"]
"""
import logging

from backend.pipelines.agentic.services import prompts
from backend.pipelines.agentic.services.agent_pool import get_main_agent, list_roles, load_agent_cards
from backend.utils.goa_configs import GENERALIST_ROLE, MAX_K, NODE_SAMPLER_AGENT

logger = logging.getLogger(__name__)


def format_agent_cards() -> str:
    """Write every model card as a short block the node sampler can read."""
    blocks = []
    for card in load_agent_cards():
        title = f"- role: {card['role']}" + (" (generalist)" if card.get("generalist") else "")
        blocks.append(
            f"{title}\n"
            f"  domain: {card['domain']}\n"
            f"  task specialization: {card['task_specialization']}\n"
            f"  special features: {card['special_features']}"
        )
    return "\n".join(blocks)


def validate_selection(selected: list[str], k: int) -> list[str]:
    """
    Make sure we end up with exactly k different, existing agents.

    The LLM may return an unknown role, a duplicate, or too few/many agents. We:
      1. keep known roles in the LLM's order and drop duplicates;
      2. cut the list to k;
      3. if it is too short, add the generalist first, then the other agents in card order.
    """
    known_roles = list_roles()

    agents = []
    for role in selected:
        if role in known_roles and role not in agents:
            agents.append(role)
    agents = agents[:k]

    for role in [GENERALIST_ROLE] + known_roles:
        if len(agents) >= k:
            break
        if role not in agents:
            agents.append(role)

    return agents


def node_sampling(query: str, k: int) -> tuple[list[str], list[str]]:
    """
    Choose the k agents that will form the graph.

    Returns (selected_agents, errors). If the node sampler fails, we fall back to
    the generalist (plus other agents to reach k) and report the problem in `errors`.
    """
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}, got {k}")

    try:
        output = get_main_agent(NODE_SAMPLER_AGENT).generate_llm_response(
            prompts.node_sampling_prompt(query, k, format_agent_cards())
        )
        selected = output.get("selected_agents", [])
        errors = []
    except Exception as error:
        logger.exception("Node sampling failed")
        selected = []
        errors = [f"node sampling failed, using the fallback agents: {error}"]

    return validate_selection(selected, k), errors
