"""
Settings used by the Graph-of-Agents pipeline (backend/pipelines/agentic).

Graph-of-Agents (GoA) — Yun et al., ICLR 2026 (arXiv 2604.17148) — answers a question
with a small *graph* of specialist LLM agents:

    node sampling  ->  initial responses  ->  edge sampling  ->  message passing  ->  graph pooling
    (pick k agents)    (each agent answers)   (agents score       (source -> target,   (combine into
                                               each other)         target -> source)    one answer)

Every tunable number and every config path lives here, so the services only
describe *what* they do (same idea as backend/utils/rag_configs.py).
"""
from pathlib import Path


# =========================================================
# PATHS
# =========================================================
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

AGENT_CARDS_FILE = CONFIGS_DIR / "agent" / "agents.yaml"        # "model cards" of the hub agents
HUB_AGENTS_DIR = CONFIGS_DIR / "agent" / "hub"                  # one <role>.yaml per specialist agent
MAIN_AGENTS_DIR = CONFIGS_DIR / "agent" / "main"                # node_sampler, graph_pooler, prompt_rewriter, ...
EDGE_SCORING_SCHEMA_FILE = CONFIGS_DIR / "schema" / "main" / "edge_scoring.schema.json"

NODE_SAMPLER_AGENT = "node_sampler"     # Meta-LLM that chooses the agents (step 1)
GRAPH_POOLER_AGENT = "graph_pooler"     # Meta-LLM that synthesises the final answer (mean pooling)


# =========================================================
# NODE SAMPLING
# =========================================================
DEFAULT_K = 10                # the paper uses k = 3 for its main results
MAX_K = 40                   # node_sampler.schema.json allows at most 10 agents
GENERALIST_ROLE = "general_theologian"   # always-safe fallback agent


# =========================================================
# EDGE SAMPLING
# =========================================================
# Each agent scores the other agents' answers. The scores of one agent are normalised to
# sum to 1, and the relevance of agent j is the sum of the scores it received:
#       S_j = sum_i score(i -> j)          (so the average S_j is exactly 1)
# Agents with S_j below the threshold are removed from the graph (paper default: 0.05).
EDGE_THRESHOLD = 0.7
MAX_RESPONSES_TO_SCORE = 13  # edge_scoring.schema.json labels responses R1 .. R13
SCORING_MAX_TOKENS = 256     # a list of scores is tiny; a small budget stops a runaway (e.g. endless whitespace) quickly
SCORING_ATTEMPTS = 2         # try again once if the scores are cut off or are not valid JSON


# =========================================================
# MESSAGE PASSING
# =========================================================
# When an agent reads another agent's answer it is told how relevant that answer is.
# The label comes from the edge weight (0..1):
HIGH_RELEVANCE = 0.7         # weight > 0.7          -> "high relevance"
MODERATE_RELEVANCE = 0.4     # 0.4 < weight <= 0.7   -> "moderate relevance", otherwise "low relevance"

MAX_PEER_RESPONSE_CHARS = 1500   # a peer's answer is cut to this length before it is shown to another agent
MAX_PARALLEL_AGENTS = 8          # threads used to call the agents at the same time (vLLM batches them)


# =========================================================
# GRAPH POOLING
# =========================================================
POOLING_METHODS = ("mean", "max")
DEFAULT_POOLING = "mean"     # "mean": graph_pooler LLM synthesises all answers; "max": take the best agent's answer
