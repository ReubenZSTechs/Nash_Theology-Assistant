import operator
from typing import Annotated, Any, TypedDict


# One agent's answer, exactly as the LLM returned it (it follows that agent's JSON schema,
# e.g. configs/schema/hub/pastor.schema.json). Every schema has at least
# "reasoning", "answer" and "confidence".
AgentResponse = dict[str, Any]


class GoAState(TypedDict, total=False):
    """
    The shared "notebook" of the Graph-of-Agents graph.

    LangGraph passes this dictionary from node to node. Each node returns only the keys
    it changed and LangGraph merges them back in.

    Every stage of the algorithm writes the agents' answers under its OWN key
    (initial_responses -> forward_responses -> final_responses). Nothing is overwritten,
    so at the end we can show how each agent's answer changed, and during message passing
    a target agent can read its sources' answers and a source can read its targets'.
    """

    # ---- Input (given by the caller) ----
    query: str                      # the question, ALREADY rewritten by the prompt_rewriter
    k: int                          # how many agents to select
    threshold: float                # edge-sampling threshold tau
    pooling: str                    # "mean" or "max"
    vector_top_k: int               # retrieval: passages from the vector database
    graph_top_k: int                # retrieval: verses from the knowledge graph

    # ---- Filled by retrieve_context (shared by every agent) ----
    context: str                    # numbered [V1].. / [G1].. blocks from the RAG pipeline
    sources: list[dict[str, Any]]   # the same results as structured data

    # ---- Filled by node_sampling ----
    selected_agents: list[str]      # roles chosen by the node_sampler, e.g. ["pastor", "ethicist"]

    # ---- Filled by initial_responses ----
    initial_responses: dict[str, AgentResponse]        # role -> first answer

    # ---- Filled by edge_sampling ----
    score_matrix: dict[str, dict[str, float]]          # scorer -> {scored agent -> normalised score}
    relevance: dict[str, float]                        # role -> S_j (sum of the scores it received)
    ranking: list[str]                                 # kept agents, most relevant first
    pruned_agents: list[str]                           # agents removed because S_j < threshold
    edges: list[dict[str, Any]]                        # [{"source", "target", "weight"}, ...]

    # ---- Filled by message passing ----
    forward_responses: dict[str, AgentResponse]        # after source -> target refinement
    final_responses: dict[str, AgentResponse]          # after target -> source refinement

    # ---- Filled by graph_pooling ----
    final_answer: dict[str, Any]

    # ---- Errors ----
    # Every node may add warnings; `operator.add` concatenates the lists instead of overwriting.
    errors: Annotated[list[str], operator.add]
