from typing import Any, Dict, List, Optional, TypedDict

class LLMCallSchema(TypedDict):
    timestamp: str
    node_name: str
    model_name: str
    status: str
    streaming: bool

    prompt_tokens: Optional[int]
    generation_tokens: Optional[int]

    latency: float
    ttft: Optional[float]
    token_generation: Optional[float]

    error_message: Optional[str]


class NodeCallSchema(TypedDict):
    timestamp: str
    node_name: str
    status: str

    latency: float

    input_chars: Optional[int]
    output_chars: Optional[int]

    error_message: Optional[str]


class EvaluationRecord(TypedDict):
    timestamp: str
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]

    scores: Dict[str, float]

    total_latency_s: Optional[float]
    node_latencies_ms: Dict[str, float]
    ttft_s: Optional[float]
    total_generation_tokens: Optional[int]

    error: Optional[str]
    metadata: Dict[str, Any]