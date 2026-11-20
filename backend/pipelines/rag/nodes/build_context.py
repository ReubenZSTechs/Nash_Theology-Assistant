"""
Node 3 — build the final context.

Turns the results of both branches into:
  * `context` — numbered text blocks that can be pasted straight into an LLM prompt.
                [V1], [V2], ... come from the vector database (sermons, Westminster),
                [G1], [G2], ... come from the knowledge graph (Bible verses).
                The LLM can cite these labels in its answer.
  * `sources` — the same items as structured data, for the API / frontend.
"""
from backend.core.decorators.log_node_call import log_node
from backend.pipelines.rag.states import RAGState
from backend.utils.rag_configs import MAX_PASSAGE_CHARS

SOURCE_NAMES = {"sermon": "Sermon", "westminster": "Westminster Standards", "bible": "Bible"}


def shorten(text: str, limit: int = MAX_PASSAGE_CHARS) -> str:
    """Cut long passages at a word boundary so the context stays small."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def format_vector_block(label: str, result: dict) -> str:
    header = f"[{label}] {SOURCE_NAMES[result['source']]} — {result['title']}"
    if result.get("url"):
        header += f" ({result['url']})"
    return f"{header}\n{shorten(result['text'])}"


def format_graph_block(label: str, result: dict) -> str:
    header = f"[{label}] Bible — {result['reference']}"
    if result.get("pericope"):
        header += f" ({result['pericope']})"
    lines = [header, f"Relation: {result['relation']}", f"BSB: {shorten(result['text'])}"]
    if result.get("text_tb"):
        lines.append(f"TB: {shorten(result['text_tb'])}")
    if result.get("linked_sources"):
        cited_by = "; ".join(item["title"] for item in result["linked_sources"])
        lines.append(f"Cited by: {cited_by}")
    return "\n".join(lines)


def build_context_and_sources(vector_results: list[dict], graph_results: list[dict]) -> tuple[str, list[dict]]:
    blocks, sources = [], []

    for number, result in enumerate(vector_results, start=1):
        label = f"V{number}"
        blocks.append(format_vector_block(label, result))
        sources.append({"label": label, **result})

    for number, result in enumerate(graph_results, start=1):
        label = f"G{number}"
        blocks.append(format_graph_block(label, result))
        sources.append({"label": label, **result})

    if not blocks:
        return "No relevant passages were found.", []

    return "\n\n".join(blocks), sources


@log_node("rag.build_context")
def build_context(state: RAGState) -> dict:
    context, sources = build_context_and_sources(state.get("vector_results", []), state.get("graph_results", []))
    return {"context": context, "sources": sources}
