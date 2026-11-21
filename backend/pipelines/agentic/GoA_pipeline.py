"""
Graph-of-Agents (GoA) pipeline — several specialist theology agents answer one question together.

Based on "Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration"
(Yun et al., ICLR 2026, arXiv 2604.17148). The agents are the NODES of a graph, and how
much they trust each other's answers decides the EDGES:

    retrieve_context      shared RAG context for every agent (the retrieval tool)
          |
    node_sampling         Meta-LLM picks the k best agents from their model cards
          |
    initial_responses     every chosen agent answers on its own (in parallel)
          |
    edge_sampling         agents score each other -> relevance S_j, pruning, source->target edges
          |
    source_to_target      targets refine using their sources' answers
          |
    target_to_source      sources finalise using their targets' refined answers
          |
    graph_pooling         mean (Meta-LLM synthesis) or max (best agent) -> one final answer

The input must be a question that the prompt_rewriter has ALREADY made self-contained
(backend/services/prompt_rewrite_service.py). The LLM reaches this pipeline through the
tool in backend/configs/tools/run_GoA.py.

Demo (run from the repository root):
    python -m backend.pipelines.agentic.GoA_pipeline "What does that verse teach?" --history history.json
"""
import argparse
import json
from functools import lru_cache
from time import perf_counter
from typing import Callable, Literal, Optional

from fastapi import APIRouter, HTTPException
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from backend.configs.tools.retrieval import execute_retrieval_tool
from backend.core.decorators.log_node_call import log_node
from backend.pipelines.agentic.services.agent_pool import get_hub_agent, get_main_agent, list_roles, load_agent_cards
from backend.pipelines.agentic.services.edge_sampling_service import edge_sampling
from backend.pipelines.agentic.services.graph_pooling_service import graph_pooling
from backend.pipelines.agentic.services.message_passing_service import pass_source_to_target, pass_target_to_source
from backend.pipelines.agentic.services.node_sampling_service import node_sampling
from backend.pipelines.agentic.services.response_service import generate_initial_responses
from backend.pipelines.agentic.states import GoAState
from backend.pipelines.rag.rag_pipeline import check_top_k, health as rag_health
from backend.services.prompt_rewrite_service import rewrite_prompt
from backend.utils.goa_configs import (
    DEFAULT_K,
    DEFAULT_POOLING,
    EDGE_THRESHOLD,
    GRAPH_POOLER_AGENT,
    MAX_K,
    NODE_SAMPLER_AGENT,
    POOLING_METHODS,
)
from backend.utils.rag_configs import DEFAULT_TOP_K, MAX_TOP_K

NO_CONTEXT = "No relevant passages were found."


# =========================================================
# GoA NODES
# Each node reads what it needs from the state, calls one service, and returns only the
# keys it changed. The algorithm itself lives in pipelines/agentic/services/.
# =========================================================
@log_node("goa.retrieve_context")
def retrieve_context_node(state: GoAState) -> dict:
    """Search the knowledge base ONCE with the retrieval tool; every agent shares this context."""
    if state["vector_top_k"] == 0 and state["graph_top_k"] == 0:
        return {"context": NO_CONTEXT, "sources": [], "errors": ["retrieval skipped (both top_k are 0)"]}

    result = execute_retrieval_tool({
        "query": state["query"],
        "vector_top_k": state["vector_top_k"],
        "graph_top_k": state["graph_top_k"],
    })
    if "error" in result:
        return {"context": NO_CONTEXT, "sources": [], "errors": [f"retrieval failed, agents answer without context: {result['error']}"]}
    return {"context": result["context"], "sources": result["sources"], "errors": result["errors"]}


@log_node("goa.node_sampling")
def node_sampling_node(state: GoAState) -> dict:
    selected_agents, errors = node_sampling(state["query"], state["k"])
    return {"selected_agents": selected_agents, "errors": errors}


@log_node("goa.initial_responses")
def initial_responses_node(state: GoAState) -> dict:
    responses, errors = generate_initial_responses(state["selected_agents"], state["query"], state["context"])
    if not responses:
        raise RuntimeError("every selected agent failed to answer: " + "; ".join(errors))
    return {"initial_responses": responses, "errors": errors}


@log_node("goa.edge_sampling")
def edge_sampling_node(state: GoAState) -> dict:
    graph = edge_sampling(state["query"], state["initial_responses"], state["threshold"])
    return graph   # score_matrix, relevance, ranking, pruned_agents, edges, errors


@log_node("goa.source_to_target")
def source_to_target_node(state: GoAState) -> dict:
    forward, errors = pass_source_to_target(
        state["query"], state["context"], state["initial_responses"], state["edges"], state["ranking"]
    )
    return {"forward_responses": forward, "errors": errors}


@log_node("goa.target_to_source")
def target_to_source_node(state: GoAState) -> dict:
    final, errors = pass_target_to_source(
        state["query"], state["context"], state["forward_responses"], state["edges"], state["ranking"]
    )
    return {"final_responses": final, "errors": errors}


@log_node("goa.graph_pooling")
def graph_pooling_node(state: GoAState) -> dict:
    final_answer, errors = graph_pooling(
        state["query"], state["final_responses"], state["relevance"], state["ranking"],
        algorithm=state["pooling"], context=state["context"],
    )
    return {"final_answer": final_answer, "errors": errors}


# =========================================================
# GoA GRAPH
# =========================================================
@lru_cache(maxsize=1)
def build_goa_graph():
    """
    Wire the nodes together and compile the graph (done once, then cached).

    The graph is a straight line: every step needs the result of the step before it.
    The parallel work (several agents at once) happens *inside* each node.
    """
    graph = StateGraph(GoAState)

    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("node_sampling", node_sampling_node)
    graph.add_node("initial_responses", initial_responses_node)
    graph.add_node("edge_sampling", edge_sampling_node)
    graph.add_node("source_to_target", source_to_target_node)
    graph.add_node("target_to_source", target_to_source_node)
    graph.add_node("graph_pooling", graph_pooling_node)

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "node_sampling")
    graph.add_edge("node_sampling", "initial_responses")
    graph.add_edge("initial_responses", "edge_sampling")
    graph.add_edge("edge_sampling", "source_to_target")
    graph.add_edge("source_to_target", "target_to_source")
    graph.add_edge("target_to_source", "graph_pooling")
    graph.add_edge("graph_pooling", END)

    return graph.compile()


# =========================================================
# GoA SERVICES
# =========================================================
def check_inputs(query: str, k: int, threshold: float, pooling: str, vector_top_k: int, graph_top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise ValueError(f"k must be an integer between 1 and {MAX_K}, got {k!r}")
    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError(f"threshold must be a number >= 0, got {threshold!r}")
    if pooling not in POOLING_METHODS:
        raise ValueError(f"pooling must be one of {POOLING_METHODS}, got {pooling!r}")
    check_top_k("vector_top_k", vector_top_k)
    check_top_k("graph_top_k", graph_top_k)


def collect_agent_outputs(state: dict) -> dict:
    """
    Group every answer by agent so you can follow one agent through the algorithm:
        {role: {"relevance", "pruned", "initial", "source_to_target", "target_to_source"}}
    Pruned agents only have an initial answer.
    """
    pruned = set(state.get("pruned_agents", []))
    return {
        role: {
            "relevance": state.get("relevance", {}).get(role),
            "pruned": role in pruned,
            "initial": initial,
            "source_to_target": state.get("forward_responses", {}).get(role),
            "target_to_source": state.get("final_responses", {}).get(role),
        }
        for role, initial in state.get("initial_responses", {}).items()
    }


def run_goa(rewritten_query: str, k: int = DEFAULT_K, threshold: float = EDGE_THRESHOLD, pooling: str = DEFAULT_POOLING,
            vector_top_k: int = DEFAULT_TOP_K, graph_top_k: int = DEFAULT_TOP_K,
            on_step: Optional[Callable[[str, dict, float], None]] = None) -> dict:
    """
    Run the whole Graph-of-Agents pipeline for one (already rewritten) question.

    Args:
        rewritten_query: self-contained question produced by the prompt_rewriter.
        k:               number of agents to select (1..MAX_K).
        threshold:       agents with relevance S_j below this are pruned (paper: 0.05).
        pooling:         "mean" (graph_pooler synthesis) or "max" (best agent's answer).
        vector_top_k / graph_top_k: size of the shared retrieval context (0 = skip that database).
        on_step:         optional callback(node_name, node_update, seconds), called after each node.
                         The demo uses it to print progress.

    Returns a dict with: query, final_answer, selected_agents, ranking, pruned_agents, relevance,
    score_matrix, edges, agent_outputs, context, sources, timings, latency, errors.
    """
    check_inputs(rewritten_query, k, threshold, pooling, vector_top_k, graph_top_k)
    start_time = perf_counter()

    initial_state = {
        "query": rewritten_query.strip(),
        "k": k,
        "threshold": float(threshold),
        "pooling": pooling,
        "vector_top_k": vector_top_k,
        "graph_top_k": graph_top_k,
        "errors": [],
    }

    # stream() instead of invoke(): "updates" tells us when each node finishes (for timings),
    # "values" gives the full state after each step; the last one is the final state.
    final_state, timings, step_start = initial_state, {}, perf_counter()
    for mode, chunk in build_goa_graph().stream(initial_state, stream_mode=["updates", "values"]):
        if mode == "values":
            final_state = chunk
            continue
        for node_name, update in chunk.items():
            seconds = round(perf_counter() - step_start, 2)
            timings[node_name] = seconds
            if on_step:
                on_step(node_name, update or {}, seconds)
        step_start = perf_counter()

    return {
        "query": final_state["query"],
        "final_answer": final_state["final_answer"],
        "selected_agents": final_state["selected_agents"],
        "ranking": final_state["ranking"],
        "pruned_agents": final_state["pruned_agents"],
        "relevance": final_state["relevance"],
        "score_matrix": final_state["score_matrix"],
        "edges": final_state["edges"],
        "agent_outputs": collect_agent_outputs(final_state),
        "context": final_state["context"],
        "sources": final_state["sources"],
        "timings": timings,
        "latency": round(perf_counter() - start_time, 2),
        "errors": final_state.get("errors", []),
    }


def list_agents() -> list[dict]:
    """The model cards of every agent the node sampler can choose from."""
    return list(load_agent_cards())


def check_component(check) -> str:
    """Run one health check and return "ok" or the error message."""
    try:
        check()
        return "ok"
    except Exception as error:
        return f"error: {error}"


def check_llm_server() -> None:
    agent = get_main_agent(NODE_SAMPLER_AGENT)
    served = [model.id for model in agent.client.models.list().data]
    if agent.agent_model not in served:
        raise RuntimeError(f"model '{agent.agent_model}' not served at {agent.agent_url}; found {served}")


def check_agent_configs() -> None:
    """Every hub and main agent config (and its schema) must load."""
    for role in list_roles():
        get_hub_agent(role)
    get_main_agent(NODE_SAMPLER_AGENT)
    get_main_agent(GRAPH_POOLER_AGENT)


def health() -> dict:
    components = {
        "llm": check_component(check_llm_server),
        "agent_configs": check_component(check_agent_configs),
        "retrieval": rag_health()["status"],
    }
    status = "ok" if all(value == "ok" for value in components.values()) else "degraded"
    return {"status": status, "components": components}


# =========================================================
# Endpoints
# =========================================================
class GoARunRequest(BaseModel):
    query: str = Field(..., min_length=1, description="A self-contained question, already rewritten by the prompt_rewriter.")
    k: int = Field(DEFAULT_K, ge=1, le=MAX_K, description="Number of agents to select.")
    threshold: float = Field(EDGE_THRESHOLD, ge=0.0, description="Prune agents whose relevance is below this.")
    pooling: Literal["mean", "max"] = Field(DEFAULT_POOLING, description="How to combine the agents' answers.")
    vector_top_k: int = Field(DEFAULT_TOP_K, ge=0, le=MAX_TOP_K, description="Shared context: passages from the vector database.")
    graph_top_k: int = Field(DEFAULT_TOP_K, ge=0, le=MAX_TOP_K, description="Shared context: verses from the knowledge graph.")


class GoARunResponse(BaseModel):
    query: str
    final_answer: dict
    selected_agents: list[str]
    ranking: list[str]
    pruned_agents: list[str]
    relevance: dict[str, float]
    score_matrix: dict[str, dict[str, float]]
    edges: list[dict]
    agent_outputs: dict[str, dict]
    context: str
    sources: list[dict]
    timings: dict[str, float]
    latency: float
    errors: list[str]


router = APIRouter(prefix="/goa", tags=["goa"])


def run_service(service, *args, **kwargs):
    """Call a service function and translate Python errors into HTTP errors."""
    try:
        return service(*args, **kwargs)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"GoA pipeline failed: {error}") from error


@router.post("/run", response_model=GoARunResponse)
def run_endpoint(request: GoARunRequest):
    return run_service(run_goa, request.query, request.k, request.threshold, request.pooling,
                       request.vector_top_k, request.graph_top_k)


@router.get("/agents")
def agents_endpoint():
    return list_agents()


@router.get("/health")
def health_endpoint():
    return health()


# =========================================================
# Demo purposes
# Full walk-through: chat history -> prompt rewriter -> retrieval (RAG) -> Graph-of-Agents
# =========================================================
def print_section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def print_progress(node_name: str, update: dict, seconds: float) -> None:
    print(f"  done: {node_name:<18} {seconds:>6.2f}s")


def print_references(answer: dict) -> None:
    for item in answer.get("scripture_references") or []:
        print(f"  - {item.get('reference')}: {item.get('note', '')}")
    for item in answer.get("confessional_references") or []:
        print(f"  - {item.get('document')} {item.get('locator')}: {item.get('note', '')}")


def print_score_matrix(score_matrix: dict, roles: list[str]) -> None:
    """Rows = the agent giving the score, columns = the agent being scored."""
    width = max(len(role) for role in roles) + 2
    print("scorer \\ scored".ljust(width) + "".join(role[:12].rjust(14) for role in roles))
    for scorer in roles:
        row = score_matrix.get(scorer, {})
        cells = "".join((f"{row[role]:.2f}" if role in row else "-").rjust(14) for role in roles)
        print(scorer.ljust(width) + cells)


def load_history(path: Optional[str]) -> list[dict]:
    """Read a chat history JSON file: [{"role": "user"|"assistant", "content": "..."}, ...]."""
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def demo(message: str, history: list[dict], k: int = DEFAULT_K, pooling: str = DEFAULT_POOLING,
         threshold: float = EDGE_THRESHOLD, vector_top_k: int = DEFAULT_TOP_K, graph_top_k: int = DEFAULT_TOP_K) -> dict:
    # 1. Prompt rewriter: the GoA tool may only run on a self-contained question.
    print_section("1. PROMPT REWRITER")
    rewrite = rewrite_prompt(message, history)
    print(f"History turns : {len(history)}")
    print(f"Original      : {rewrite['original_message']}")
    print(f"Rewritten     : {rewrite['rewritten_query']}")
    print(f"Follow-up?    : {rewrite['is_follow_up']}")
    if rewrite["error"]:
        print(f"[warning] {rewrite['error']}")

    # This is the tool call the main LLM would make (see backend/configs/tools/run_GoA.py).
    tool_call = {"rewritten_query": rewrite["rewritten_query"], "k": k, "pooling": pooling}
    print(f"\nTool call     : run_graph_of_agents({json.dumps(tool_call, ensure_ascii=False)})")

    # 2. Run the Graph-of-Agents pipeline.
    print_section("2. RUNNING THE GRAPH-OF-AGENTS PIPELINE")
    result = run_goa(rewrite["rewritten_query"], k, threshold, pooling, vector_top_k, graph_top_k, on_step=print_progress)

    print_section("3. SHARED RETRIEVAL CONTEXT (RAG)")
    for source in result["sources"]:
        title = source.get("reference") or source.get("title")
        relation = f" — {source['relation']}" if source.get("relation") else ""
        print(f"[{source['label']}] {source.get('source', 'bible')}: {title}{relation}")
    if not result["sources"]:
        print(result["context"])

    print_section("4. NODE SAMPLING — selected agents")
    for role in result["selected_agents"]:
        print(f"  - {role}")

    print_section("5. INITIAL RESPONSES")
    for role, outputs in result["agent_outputs"].items():
        initial = outputs["initial"]
        print(f"\n[{role}] (confidence {initial.get('confidence')})\n{initial.get('answer', '')}")

    print_section("6. EDGE SAMPLING — scores, relevance and edges")
    roles = list(result["agent_outputs"])
    if result["score_matrix"]:
        print_score_matrix(result["score_matrix"], roles)
    print("\nRelevance S_j (sum of the scores received, average = 1):")
    for role in sorted(result["relevance"], key=result["relevance"].get, reverse=True):
        status = "PRUNED" if role in result["pruned_agents"] else "kept"
        print(f"  {role:<24} {result['relevance'][role]:.3f}  {status}")
    print("\nEdges (source -> target, weight):")
    for edge in result["edges"]:
        print(f"  {edge['source']} -> {edge['target']}  ({edge['weight']:.2f})")
    if not result["edges"]:
        print("  (none: only one agent in the graph)")

    print_section("7. MESSAGE PASSING — how each answer changed")
    for role in result["ranking"]:
        outputs = result["agent_outputs"][role]
        print(f"\n[{role}]")
        print(f"  source -> target : {outputs['source_to_target'].get('answer', '')}")
        print(f"  target -> source : {outputs['target_to_source'].get('answer', '')}")

    print_section(f"8. FINAL ANSWER ({result['final_answer']['method']} pooling)")
    final_answer = result["final_answer"]
    print(final_answer["answer"])
    print(f"\nIntegrated agents: {', '.join(final_answer['integrated_agents'])}")
    print(f"Confidence       : {final_answer.get('confidence')}")
    print("References:")
    print_references(final_answer)
    for disagreement in final_answer.get("unresolved_disagreements") or []:
        print(f"Unresolved: {disagreement}")

    print_section("9. TIMINGS AND WARNINGS")
    for node_name, seconds in result["timings"].items():
        print(f"  {node_name:<18} {seconds:>6.2f}s")
    print(f"  {'total':<18} {result['latency']:>6.2f}s")
    for error in result["errors"]:
        print(f"[warning] {error}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Demo: prompt rewriter + RAG + Graph-of-Agents for one message")
    parser.add_argument("message", help="the user's latest chat message")
    parser.add_argument("--history", help='JSON file with earlier turns: [{"role": "user", "content": "..."}, ...]')
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"number of agents (1-{MAX_K})")
    parser.add_argument("--pooling", choices=POOLING_METHODS, default=DEFAULT_POOLING)
    parser.add_argument("--threshold", type=float, default=EDGE_THRESHOLD, help="edge-sampling pruning threshold")
    parser.add_argument("--vector-top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--graph-top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--save", help="also write the full result to this JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo_result = demo(args.message, load_history(args.history), args.k, args.pooling,
                       args.threshold, args.vector_top_k, args.graph_top_k)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as output_file:
            json.dump(demo_result, output_file, ensure_ascii=False, indent=2, default=str)
        print(f"\nFull result written to {args.save}")
