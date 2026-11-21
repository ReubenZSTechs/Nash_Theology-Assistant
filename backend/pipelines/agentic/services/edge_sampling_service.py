"""
Step 3 of Graph-of-Agents — edge sampling.

After every agent has answered, the agents judge each other. This tells us which answers
are the most trustworthy and therefore which way information should flow in the graph.

  1. Scoring     every agent i scores every OTHER agent j's answer (0..1). An agent never
                 scores itself, to avoid self-bias.
  2. Normalising the scores of one scorer are divided by their sum, so each row sums to 1.
                 Now every agent has the same "voting power".
  3. Relevance   S_j = sum_i score(i -> j)        (paper, Eq. 2)
                 = how much the other agents trust agent j. The average S_j is exactly 1.
  4. Pruning     agents with S_j < threshold are removed from the graph.
  5. Edges       the kept agents are sorted by S (most relevant first). A more relevant agent
                 is a SOURCE for every less relevant agent (its TARGETS):
                     ranking = [A, B, C]   ->   edges A->B, A->C, B->C
                 The weight of edge i -> j says how much j should listen to i compared with
                 j's other sources (paper, Eq. 3):
                     A_ij = S_i / sum_{k in sources(j)} S_k     (weights into j sum to 1)

Small example with 3 agents (rows = scorer, already normalised):
                 A     B     C
          A      -    0.6   0.4
          B     0.7    -    0.3
          C     0.5   0.5    -
      S_A = 0.7 + 0.5 = 1.2,  S_B = 0.6 + 0.5 = 1.1,  S_C = 0.4 + 0.3 = 0.7
      ranking = [A, B, C]; edges A->B (1.0), A->C (1.2/2.3 = 0.52), B->C (1.1/2.3 = 0.48)
"""
import json
import logging
from functools import lru_cache

from backend.pipelines.agentic.services import prompts
from backend.pipelines.agentic.services.agent_pool import format_response_for_peers, get_hub_agent, run_in_parallel
from backend.utils.goa_configs import (
    EDGE_SCORING_SCHEMA_FILE,
    EDGE_THRESHOLD,
    MAX_RESPONSES_TO_SCORE,
    SCORING_ATTEMPTS,
    SCORING_MAX_TOKENS,
)
from backend.utils.read_agent_schema import read_schema

logger = logging.getLogger(__name__)


# =========================================================
# 1. SCORING
# =========================================================
@lru_cache(maxsize=1)
def edge_scoring_response_format() -> dict:
    """
    The structured-output format for scoring: {"scores": [{"response_id": "R1", "score": 0.8}, ...]}.

    The hub agents normally answer in their own schema. For scoring we pass this format
    through `extra_args`, which replaces the agent's usual `response_format` for that one call
    (LLMAgent.format_generation_payload copies extra_args over the request).
    """
    schema = read_schema(EDGE_SCORING_SCHEMA_FILE)
    if not schema:
        raise FileNotFoundError(f"Edge scoring schema could not be read: {EDGE_SCORING_SCHEMA_FILE}")
    return {"type": "json_schema", "json_schema": {"name": "edge_scoring", "schema": schema, "strict": True}}


def request_scores(scorer: str, user_prompt: str) -> dict:
    """
    Ask one agent for its scores, with a small token budget.

    We call `request_completion` directly (instead of `generate_llm_response`) so we can set
    max_tokens ourselves: the agent's normal budget (2048 tokens) is meant for long answers,
    and a broken scoring reply would otherwise run on for half a minute before being cut.
    """
    agent = get_hub_agent(scorer)
    for _ in range(SCORING_ATTEMPTS):
        response = agent.request_completion(
            user_prompt,
            extra_args={"response_format": edge_scoring_response_format(), "max_tokens": SCORING_MAX_TOKENS},
        )
        if agent.last_finish_reason == "length":
            continue                                  # cut off -> try again
        try:
            return agent.parse_output(response.choices[0].message.content)
        except json.JSONDecodeError:
            continue                                  # not valid JSON -> try again
    raise ValueError(f"no valid scores after {SCORING_ATTEMPTS} attempts")


def score_peer_responses(scorer: str, query: str, responses: dict[str, dict]) -> dict[str, float]:
    """
    Let one agent score the answers of all the other agents.

    The other answers are shown anonymously as R1, R2, ... so the scorer judges the content,
    not the name of the agent. Returns {other_role: raw score between 0 and 1}.
    """
    others = [role for role in responses if role != scorer]
    label_to_role = {f"R{number}": role for number, role in enumerate(others, start=1)}

    labelled = "\n\n".join(
        f"[{label}]\n{format_response_for_peers(responses[role])}" for label, role in label_to_role.items()
    )
    output = request_scores(scorer, prompts.edge_scoring_prompt(query, labelled, list(label_to_role)))

    scores = {role: 0.0 for role in others}          # an answer the scorer skipped gets 0
    for item in output.get("scores", []):
        role = label_to_role.get(item.get("response_id"))
        if role is not None:
            scores[role] = min(max(float(item.get("score", 0.0)), 0.0), 1.0)
    return scores


def collect_scores(query: str, responses: dict[str, dict]) -> tuple[dict[str, dict[str, float]], list[str]]:
    """
    Every agent scores the others, in parallel. Returns (raw_score_matrix, errors).

    If one scorer fails it gives every other agent the same score, so it does not
    favour anyone (after normalising its row is uniform).
    """
    results = run_in_parallel(lambda scorer: score_peer_responses(scorer, query, responses), list(responses))

    matrix, errors = {}, []
    for scorer, result in results.items():
        if isinstance(result, Exception):
            logger.warning("Agent %s failed to score the others: %s", scorer, result)
            errors.append(f"agent '{scorer}' failed to score the other responses (using equal scores): {result}")
            matrix[scorer] = {role: 1.0 for role in responses if role != scorer}
        else:
            matrix[scorer] = result
    return matrix, errors


# =========================================================
# 2-5. FROM SCORES TO A GRAPH (pure functions, no LLM)
# =========================================================
def normalize_scores(matrix: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Divide each scorer's scores by their sum so every row sums to 1 (all zeros -> equal scores)."""
    normalized = {}
    for scorer, row in matrix.items():
        total = sum(row.values())
        if total > 0:
            normalized[scorer] = {role: score / total for role, score in row.items()}
        else:
            normalized[scorer] = {role: 1.0 / len(row) for role in row} if row else {}
    return normalized


def compute_relevance(matrix: dict[str, dict[str, float]], roles: list[str]) -> dict[str, float]:
    """S_j = sum of the (normalised) scores that agent j received from the other agents."""
    relevance = {role: 0.0 for role in roles}
    for row in matrix.values():
        for role, score in row.items():
            relevance[role] += score
    return relevance


def prune_agents(relevance: dict[str, float], roles: list[str], threshold: float) -> tuple[list[str], list[str]]:
    """
    Sort the agents by relevance (most relevant first) and drop those below the threshold.

    `roles` gives the tie-break order (the node sampler's order). The most relevant agent is
    always kept, so the graph is never empty. Returns (ranking, pruned_agents).
    """
    ordered = sorted(roles, key=lambda role: (-relevance[role], roles.index(role)))
    ranking = [role for role in ordered if relevance[role] >= threshold] or ordered[:1]
    pruned = [role for role in ordered if role not in ranking]
    return ranking, pruned


def build_edges(ranking: list[str], relevance: dict[str, float]) -> list[dict]:
    """
    Connect every agent to every less relevant agent: source (higher S) -> target (lower S).

    Weight of i -> j: A_ij = S_i / (sum of S over all sources of j), so the weights going
    into one target add up to 1.
    """
    edges = []
    for position, target in enumerate(ranking):
        sources = ranking[:position]                     # everyone ranked above the target
        total = sum(relevance[source] for source in sources)
        for source in sources:
            weight = relevance[source] / total if total > 0 else 1.0 / len(sources)
            edges.append({"source": source, "target": target, "weight": round(weight, 4)})
    return edges


# =========================================================
# THE WHOLE STEP
# =========================================================
def edge_sampling(query: str, responses: dict[str, dict], threshold: float = EDGE_THRESHOLD) -> dict:
    """
    Build the agent graph from the initial responses.

    Returns {"score_matrix", "relevance", "ranking", "pruned_agents", "edges", "errors"}.
    With only one agent there is nobody to score, so the graph is that single node.
    """
    roles = list(responses)
    if len(roles) > MAX_RESPONSES_TO_SCORE + 1:
        raise ValueError(f"edge sampling supports at most {MAX_RESPONSES_TO_SCORE + 1} agents, got {len(roles)}")

    if len(roles) == 1:
        return {"score_matrix": {}, "relevance": {roles[0]: 1.0}, "ranking": roles,
                "pruned_agents": [], "edges": [], "errors": []}

    raw_matrix, errors = collect_scores(query, responses)
    matrix = normalize_scores(raw_matrix)
    relevance = compute_relevance(matrix, roles)
    ranking, pruned = prune_agents(relevance, roles, threshold)

    return {
        "score_matrix": {scorer: {role: round(score, 4) for role, score in row.items()} for scorer, row in matrix.items()},
        "relevance": {role: round(score, 4) for role, score in relevance.items()},
        "ranking": ranking,
        "pruned_agents": pruned,
        "edges": build_edges(ranking, relevance),
        "errors": errors,
    }
