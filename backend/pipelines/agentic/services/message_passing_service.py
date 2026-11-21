"""
Step 4 of Graph-of-Agents — message passing.

Once the graph exists, the agents improve their answers by reading their neighbours'
answers. This happens in two directions, one after the other:

  1. Source -> target (paper, Eq. 4)
     Every TARGET reads the initial answers of its SOURCES (the more relevant agents)
     and refines its own answer.          R'_j  = refine(R_j  | sources' R)

  2. Target -> source (paper, Eq. 5)
     Every SOURCE reads how its TARGETS refined their answers after seeing its answer,
     and finalises its own answer.        R''_i = refine(R'_i | targets' R')

Each neighbour's answer is shown with a relevance label taken from the edge weight
(high > 0.7, moderate > 0.4, otherwise low), most relevant first, so the agent knows how
much to trust it.

In the paper, a source refines its *initial* answer in step 2. We pass the agent's
*latest* answer instead (for the top agent this is the same thing). An agent in the middle
of the ranking is both a target and a source, so this way it keeps what it learned in step 1.
"""
import logging

from backend.pipelines.agentic.services import prompts
from backend.pipelines.agentic.services.agent_pool import format_response_for_peers, get_hub_agent, run_in_parallel
from backend.utils.goa_configs import HIGH_RELEVANCE, MODERATE_RELEVANCE

logger = logging.getLogger(__name__)


# =========================================================
# HELPERS
# =========================================================
def relevance_label(weight: float) -> str:
    """Turn an edge weight (0..1) into words the LLM understands."""
    if weight > HIGH_RELEVANCE:
        return "high relevance"
    if weight > MODERATE_RELEVANCE:
        return "moderate relevance"
    return "low relevance"


def sources_of(target: str, edges: list[dict]) -> list[dict]:
    """Edges coming INTO `target`, strongest first."""
    incoming = [edge for edge in edges if edge["target"] == target]
    return sorted(incoming, key=lambda edge: edge["weight"], reverse=True)


def targets_of(source: str, edges: list[dict]) -> list[dict]:
    """Edges going OUT of `source`, strongest first."""
    outgoing = [edge for edge in edges if edge["source"] == source]
    return sorted(outgoing, key=lambda edge: edge["weight"], reverse=True)


def format_neighbour_responses(neighbours: list[tuple[str, float]], responses: dict[str, dict]) -> str:
    """
    Write the neighbours' answers as labelled blocks, e.g.

        [systematic_theologian — high relevance (0.82)]
        Answer: ...
    """
    return "\n\n".join(
        f"[{role} — {relevance_label(weight)} ({weight:.2f})]\n{format_response_for_peers(responses[role])}"
        for role, weight in neighbours
    )


def refine_all(prompts_by_role: dict[str, str], previous: dict[str, dict], step_name: str) -> tuple[dict[str, dict], list[str]]:
    """
    Send every refinement prompt in parallel. An agent whose refinement fails keeps its
    previous answer, so the graph never loses a node. Returns (refined_responses, errors).
    """
    results = run_in_parallel(lambda role: get_hub_agent(role).generate_llm_response(prompts_by_role[role]), prompts_by_role)

    refined, errors = {}, []
    for role, result in results.items():
        if isinstance(result, Exception):
            logger.warning("Agent %s failed during %s: %s", role, step_name, result)
            errors.append(f"agent '{role}' failed during {step_name}, keeping its previous answer: {result}")
            refined[role] = previous[role]
        else:
            refined[role] = result
    return refined, errors


# =========================================================
# THE TWO DIRECTIONS
# =========================================================
def pass_source_to_target(query: str, context: str, initial_responses: dict[str, dict],
                          edges: list[dict], ranking: list[str]) -> tuple[dict[str, dict], list[str]]:
    """
    Every target refines its initial answer using its sources' initial answers.

    Agents without sources (the most relevant agent) keep their initial answer.
    Returns (forward_responses, errors) for every agent in `ranking`.
    """
    refinement_prompts = {}
    for target in ranking:
        incoming = sources_of(target, edges)
        if incoming:
            neighbours = [(edge["source"], edge["weight"]) for edge in incoming]
            refinement_prompts[target] = prompts.source_to_target_prompt(
                query,
                context,
                own_response=format_response_for_peers(initial_responses[target]),
                source_responses=format_neighbour_responses(neighbours, initial_responses),
            )

    refined, errors = refine_all(refinement_prompts, initial_responses, "source-to-target refinement")
    forward_responses = {role: refined.get(role, initial_responses[role]) for role in ranking}
    return forward_responses, errors


def pass_target_to_source(query: str, context: str, forward_responses: dict[str, dict],
                          edges: list[dict], ranking: list[str]) -> tuple[dict[str, dict], list[str]]:
    """
    Every source finalises its answer using its targets' refined answers.

    Agents without targets (the least relevant agent) keep their source-to-target answer.
    Returns (final_responses, errors) for every agent in `ranking`.
    """
    refinement_prompts = {}
    for source in ranking:
        outgoing = targets_of(source, edges)
        if outgoing:
            neighbours = [(edge["target"], edge["weight"]) for edge in outgoing]
            refinement_prompts[source] = prompts.target_to_source_prompt(
                query,
                context,
                own_response=format_response_for_peers(forward_responses[source]),
                target_responses=format_neighbour_responses(neighbours, forward_responses),
            )

    refined, errors = refine_all(refinement_prompts, forward_responses, "target-to-source refinement")
    final_responses = {role: refined.get(role, forward_responses[role]) for role in ranking}
    return final_responses, errors


def message_passing(query: str, context: str, initial_responses: dict[str, dict],
                    edges: list[dict], ranking: list[str]) -> dict:
    """
    Run both directions one after the other (handy outside LangGraph, e.g. in a notebook).

    Returns {"forward_responses", "final_responses", "errors"}.
    """
    forward, forward_errors = pass_source_to_target(query, context, initial_responses, edges, ranking)
    final, final_errors = pass_target_to_source(query, context, forward, edges, ranking)
    return {"forward_responses": forward, "final_responses": final, "errors": forward_errors + final_errors}
