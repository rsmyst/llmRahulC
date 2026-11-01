"""Dual-mode embedding provider: Gemini API or local Nomic.

Uses gemini-embedding-001 when SM_GEMINI_API_KEY is set, otherwise
loads nomic-embed-text-v1.5 locally via sentence-transformers (768-dim,
normalized so FAISS IndexFlatIP == cosine).
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from .config import Config

_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_QUERY = "RETRIEVAL_QUERY"
_MAX_BATCH = 100
_WORD_RE = re.compile(r"[a-zA-Z0-9_\-\.]+")


class EmbeddingProvider:
    def __init__(self, config: Config):
        self._config = config
        self._model = f"models/{config.embed_model}"
        self._dim = config.embed_dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def backend_name(self) -> str:
        return "gemini" if self._config.has_gemini_key else "local_nomic_embed"

    def _get_local_model(self):
        if not hasattr(self, "_local_model"):
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(
                "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
            )
        return self._local_model

    def _local_embed_text(self, text: str) -> list[float]:
        return self._local_embed_texts([text])[0]

    def _local_embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_local_model()
        embeddings = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return embeddings.tolist()

    # ------------------------------------------------------------- Gemini API

    @staticmethod
    def _headers(config: Config) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": config.gemini_api_key,
        }

    def _request(self, path: str, payload: dict) -> dict:
        import httpx

        cfg = self._config
        url = f"{cfg.gemini_base_url.rstrip('/')}/{path}"
        resp = httpx.post(
            url, json=payload, headers=self._headers(cfg), timeout=cfg.request_timeout
        )
        resp.raise_for_status()
        return resp.json()

    def _embed_payload(self, text: str, task_type: str) -> dict:
        return {
            "model": self._model,
            "content": {"parts": [{"text": text[:10000]}]},
            "taskType": task_type,
            "outputDimensionality": self._dim,
        }

    def _embed_gemini(
        self, texts: Sequence[str], task_type: str
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            chunk = list(texts[start : start + _MAX_BATCH])
            if len(chunk) == 1:
                data = self._request(
                    f"models/{self._config.embed_model}:embedContent",
                    self._embed_payload(chunk[0], task_type),
                )
                vecs = [data["embedding"]["values"]]
            else:
                data = self._request(
                    f"models/{self._config.embed_model}:batchEmbedContents",
                    {
                        "requests": [
                            self._embed_payload(t, task_type) for t in chunk
                        ]
                    },
                )
                vecs = [e["values"] for e in data["embeddings"]]

            for v in vecs:
                arr = np.asarray(v, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 0:
                    arr = arr / norm
                vectors.append(arr.tolist())
        return vectors

    # ---------------------------------------------------------- Public API

    def embed(
        self,
        texts: Sequence[str],
        task_type: str = _TASK_DOCUMENT,
    ) -> list[list[float]]:
        """Embed a list of texts, returning L2-normalized vectors."""
        if not texts:
            return []

        if self._config.has_gemini_key:
            try:
                return self._embed_gemini(texts, task_type=task_type)
            except Exception:
                # Fall back to local hashing on network/key errors
                pass

        return self._local_embed_texts(texts)

    def embed_one(
        self, text: str, task_type: str = _TASK_DOCUMENT
    ) -> list[float]:
        return self.embed([text], task_type=task_type)[0]
