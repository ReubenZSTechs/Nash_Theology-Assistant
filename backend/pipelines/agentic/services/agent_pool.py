"""
The agent pool: loading agents, calling them in parallel, and showing one agent's answer to another.

In Graph-of-Agents every specialist agent is a *node* of the graph. Here every node is an
LLMAgent (backend/core/llm/agent_init.py) built from one YAML file in configs/agent/hub/.
All agents run on the same vLLM model; what makes them different is their system prompt
and their JSON output schema.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Iterable

import yaml

from backend.core.llm.agent_init import LLMAgent
from backend.utils.goa_configs import (
    AGENT_CARDS_FILE,
    HUB_AGENTS_DIR,
    MAIN_AGENTS_DIR,
    MAX_PARALLEL_AGENTS,
    MAX_PEER_RESPONSE_CHARS,
)


# =========================================================
# LOADING AGENTS
# =========================================================
@lru_cache(maxsize=1)
def load_agent_cards() -> tuple[dict, ...]:
    """
    Read the "model cards" of the hub agents from configs/agent/agents.yaml.

    A card describes what an agent is good at (role, domain, task_specialization,
    special_features). The node sampler reads these cards to choose the agents.
    Returned as a tuple so the cached value cannot be changed by accident.
    """
    with open(AGENT_CARDS_FILE, "r", encoding="utf-8") as file:
        cards = yaml.safe_load(file)["agents"]
    return tuple(cards)


def list_roles() -> list[str]:
    """All hub agent roles, in the order of agents.yaml."""
    return [card["role"] for card in load_agent_cards()]


@lru_cache(maxsize=None)
def get_hub_agent(role: str) -> LLMAgent:
    """Build (once) the specialist agent for a role, e.g. get_hub_agent("pastor")."""
    if role not in list_roles():
        raise KeyError(f"Unknown agent role '{role}'. Known roles: {', '.join(list_roles())}")
    return LLMAgent(agent_config=HUB_AGENTS_DIR / f"{role}.yaml")


@lru_cache(maxsize=None)
def get_main_agent(name: str) -> LLMAgent:
    """Build (once) a "Meta-LLM" agent from configs/agent/main/, e.g. "node_sampler"."""
    return LLMAgent(agent_config=MAIN_AGENTS_DIR / f"{name}.yaml")


# =========================================================
# CALLING AGENTS IN PARALLEL
# =========================================================
def run_in_parallel(function: Callable[[Any], Any], items: Iterable[Any]) -> dict[Any, Any]:
    """
    Call `function(item)` for every item at the same time and return {item: result}.

    If a call raises, its result is the exception object instead, so one failing agent
    never stops the others. The caller decides what to do with failures.
    vLLM batches the parallel requests on the GPU, so this is much faster than a loop.
    """
    items = list(items)
    if not items:
        return {}

    def safe_call(item):
        try:
            return function(item)
        except Exception as error:      # noqa: BLE001 - reported to the caller, not hidden
            return error

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_AGENTS, len(items))) as pool:
        results = list(pool.map(safe_call, items))
    return dict(zip(items, results))


# =========================================================
# SHOWING ONE AGENT'S ANSWER TO ANOTHER
# =========================================================
def shorten(text: str, limit: int = MAX_PEER_RESPONSE_CHARS) -> str:
    """Cut long text at a word boundary."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def format_references(response: dict) -> str:
    """Collect the Scripture and Westminster references of a response into one line."""
    references = [item.get("reference", "") for item in response.get("scripture_references") or []]
    references += [
        f"{item.get('document', '')} {item.get('locator', '')}".strip()
        for item in response.get("confessional_references") or []
    ]
    return ", ".join(reference for reference in references if reference)


def answer_text(response: dict) -> str:
    """
    The answer of a response. If the LLM left "answer" empty but explained itself in
    "reasoning", use the reasoning instead, so other agents still have something to read.
    """
    return (response.get("answer") or "").strip() or (response.get("reasoning") or "").strip()


def format_response_for_peers(response: dict) -> str:
    """
    Turn an agent's JSON answer into short text that another agent can read.

    The hub schemas are all different (a pastor has "applications", an exegete has
    "key_terms", ...), but they all share "answer", "confidence" and usually references,
    so only those are shown. This keeps the prompts small for an 8B model.
    """
    lines = [f"Answer: {shorten(answer_text(response))}"]
    references = format_references(response)
    if references:
        lines.append(f"References: {references}")
    if "confidence" in response:
        lines.append(f"Confidence: {response['confidence']}")
    return "\n".join(lines)
