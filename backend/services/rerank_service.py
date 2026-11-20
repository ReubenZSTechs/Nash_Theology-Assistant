"""
LLM-as-reranker.

Instead of loading a separate cross-encoder model, we reuse the LLM that is already
running (vLLM, served name "llm"). For every (question, passage) pair the LLM rates the
passage on a 0-3 scale. vLLM's structured output (`choice`) forces the answer to be one
digit, and the token logprobs tell us how sure the LLM is about each digit:

    score = (0 * p(0) + 1 * p(1) + 2 * p(2) + 3 * p(3)) / 3        -> between 0 and 1

Using the probabilities (not just the chosen digit) gives smooth scores, so two passages
that are both rated "2" can still be ordered.

The prompt and generation settings live in backend/configs/agent/reranker.yaml.
"""
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI

from backend.core.exceptions.RerankExceptions import RerankServiceError
from backend.utils.read_agent_config import read_config

load_dotenv()

logger = logging.getLogger(__name__)

RERANKER_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "agent" / "reranker.yaml"
RATINGS = ("0", "1", "2", "3")
MAX_RATING = 3


def expected_relevance(top_logprobs) -> float:
    """
    Turn the logprobs of the first generated token into a 0-1 relevance score.

    `top_logprobs` is a list of objects with `.token` and `.logprob` (OpenAI format).
    Only the digit tokens are used; their probabilities are re-normalised to sum to 1.
    """
    probabilities = {}
    for candidate in top_logprobs:
        token = candidate.token.strip()
        if token in RATINGS:
            probabilities[int(token)] = probabilities.get(int(token), 0.0) + math.exp(candidate.logprob)

    total = sum(probabilities.values())
    if total == 0:
        raise RerankServiceError("LLM reranker returned no rating digit in its logprobs")

    expected_rating = sum(rating * probability for rating, probability in probabilities.items()) / total
    return expected_rating / MAX_RATING


def shorten(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


class RerankService:
    def __init__(self, config_path: Path = RERANKER_CONFIG):
        config = read_config(config_path)
        if not config:
            raise RerankServiceError(f"Reranker config not found or empty: {config_path}")

        self.system_prompt = config["system_prompt"]
        self.user_prompt = config["user_prompt"]
        self.generation = config["generation"]

        self.base_url = os.getenv("LLM_URL", "http://localhost:15000/v1")
        self.model = os.getenv("LLM_MODEL", "llm")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("VLLM_API_KEY") or "not-needed",
            timeout=self.generation.get("timeout", 30),
            max_retries=self.generation.get("max_retries", 1),
        )


    def check_server(self) -> None:
        try:
            served = [model.id for model in self.client.models.list().data]
        except APIConnectionError as error:
            raise RerankServiceError(f"LLM server unreachable at {self.base_url}") from error

        if self.model not in served:
            raise RerankServiceError(f"Model '{self.model}' not served at {self.base_url}; found {served}")


    def build_messages(self, query: str, passage: str) -> list[dict]:
        passage = shorten(passage, self.generation.get("max_passage_chars", 1500))
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(query=query, passage=passage)},
        ]


    def score(self, query: str, passage: str) -> float:
        """Relevance of one passage to the query, between 0 and 1."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.build_messages(query, passage),
                temperature=self.generation.get("temperature", 0.0),
                max_tokens=self.generation.get("max_tokens", 1),
                logprobs=True,
                top_logprobs=self.generation.get("top_logprobs", 10),
                extra_body={"structured_outputs": {"choice": list(RATINGS)}},
            )
        except APIConnectionError as error:
            raise RerankServiceError(f"LLM server unreachable at {self.base_url}") from error
        except APIStatusError as error:
            raise RerankServiceError(f"Rerank request failed ({error.status_code}): {error.message}") from error

        first_token = response.choices[0].logprobs.content[0]
        return expected_relevance(first_token.top_logprobs)


    def rerank(self, query: str, documents: list[str], top_n: Optional[int] = None) -> list[dict]:
        """
        Score every document and return them best first:
            [{"index": position in `documents`, "relevance_score": 0..1}, ...]

        The requests are sent in parallel; vLLM batches them on the GPU and the shared
        system prompt is reused through prefix caching.
        """
        if not documents:
            return []

        workers = min(self.generation.get("max_workers", 16), len(documents))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(lambda document: self.score(query, document), documents))

        ranked = sorted(
            ({"index": index, "relevance_score": round(score, 4)} for index, score in enumerate(scores)),
            key=lambda item: item["relevance_score"],
            reverse=True,
        )
        return ranked[:top_n] if top_n else ranked


@lru_cache(maxsize=1)
def get_rerank_service() -> RerankService:
    return RerankService()
