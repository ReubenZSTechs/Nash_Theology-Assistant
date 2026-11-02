import logging
import os
from collections.abc import Iterable
from functools import lru_cache
from typing import Optional, Union

import numpy as np
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI
from backend.core.exceptions.EmbeddingExceptions import EmbeddingServiceError

load_dotenv()

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, timeout: Optional[float] = None, max_retries: Optional[int] = None, check: bool = True):
        self.base_url = os.getenv("EMBEDDING_URL", "http://localhost:15001/v1")
        self.model = os.getenv("EMBEDDING_MODEL", "embedding")
        self.dimensions = int(os.getenv("EMBED_DIM", "1024"))
        self.batch_size = int(os.getenv("EMBED_BATCH", "64"))
        self.max_tokens = int(os.getenv("EMBED_MAX_TOKENS", "8192"))

        self.client = OpenAI(
            base_url=self.base_url,
            api_key="not-needed",
            timeout=timeout or float(os.getenv("EMBED_TIMEOUT", "300")),
            max_retries=max_retries if max_retries is not None else int(os.getenv("EMBED_MAX_RETRIES", "3")),
        )

        if check:
            self.check_server()


    def check_server(self) -> None:
        try:
            served = [model.id for model in self.client.models.list().data]
        except APIConnectionError as error:
            raise EmbeddingServiceError(f"Embedding server unreachable at {self.base_url}") from error

        if self.model not in served:
            raise EmbeddingServiceError(f"Model '{self.model}' not served at {self.base_url}; found {served}")

        logger.info(f"Embedding service ready: {self.model} via {self.base_url}")


    def encode(self, texts: Union[str, Iterable[str]]) -> np.ndarray:
        if isinstance(texts, str):
            return self.encode_batch([texts])[0]

        return self.encode_batch(texts)


    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError(f"embed_query expects a string, got {type(text).__name__}")

        return self.encode(text).tolist()


    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return self.encode_batch(texts).tolist()


    def encode_batch(self, texts: Iterable[str]) -> np.ndarray:
        items = self.validate(texts)
        if not items:
            return np.empty((0, self.dimensions), dtype=np.float32)

        vectors = []
        for start in range(0, len(items), self.batch_size):
            vectors.extend(self.request(items[start:start + self.batch_size]))

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (len(items), self.dimensions):
            raise EmbeddingServiceError(
                f"Expected {len(items)} x {self.dimensions} embeddings, got shape {matrix.shape}"
            )

        return self.normalize(matrix)


    def validate(self, texts: Iterable[str]) -> list[str]:
        if not isinstance(texts, Iterable):
            raise TypeError(f"texts must be a string or an iterable of strings, got {type(texts).__name__}")

        items = list(texts)
        invalid = [index for index, text in enumerate(items) if not isinstance(text, str)]
        if invalid:
            raise TypeError(f"All inputs must be strings; non-string values at positions {invalid[:10]}")

        blank = [index for index, text in enumerate(items) if not text.strip()]
        if blank:
            raise ValueError(f"Empty or whitespace-only text at positions {blank[:10]}")

        return items


    def request(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float",
                extra_body={"truncate_prompt_tokens": self.max_tokens},
            )
        except APIConnectionError as error:
            raise EmbeddingServiceError(f"Embedding server unreachable at {self.base_url}") from error
        except APIStatusError as error:
            raise EmbeddingServiceError(f"Embedding request failed ({error.status_code}): {error.message}") from error

        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]


    @staticmethod
    def normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()