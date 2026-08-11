"""GeminiEmbedder — optional remote embedder using Google's google-genai SDK.

See docs/design/02-lld-memlayer.md §3. Implements the same `EmbeddingBase`
interface as `SentenceTransformerEmbedder`, proving the plug-in abstraction
holds for a second real provider.
"""

from __future__ import annotations

import os

from google import genai

from memlayer.embeddings.base import EmbeddingBase

DEFAULT_MODEL = "text-embedding-004"
DEFAULT_DIMS = 768
API_KEY_ENV_VAR = "GEMINI_API_KEY"


class GeminiEmbedder(EmbeddingBase):
    """Optional embedder backed by Gemini's `text-embedding-004` model (768-dim).

    API key resolution, in order: constructor arg, then the `GEMINI_API_KEY`
    environment variable, else raise `ValueError` immediately — fail fast at
    construction time rather than at first use.
    """

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        resolved_api_key = api_key or os.environ.get(API_KEY_ENV_VAR)
        if not resolved_api_key:
            raise ValueError(
                "GeminiEmbedder requires an API key: pass api_key=... or set the "
                f"{API_KEY_ENV_VAR} environment variable."
            )
        self.model = model
        self.dims = DEFAULT_DIMS
        self._client = genai.Client(api_key=resolved_api_key)

    def embed(self, text: str) -> list[float]:
        response = self._client.models.embed_content(model=self.model, contents=text)
        if not response.embeddings:
            raise RuntimeError(
                f"GeminiEmbedder got an empty embeddings response for model {self.model!r}."
            )
        return list(response.embeddings[0].values)
