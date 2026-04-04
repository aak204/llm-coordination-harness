from __future__ import annotations

from coord_harness.config.models import ModelSpec
from coord_harness.core.enums import ModelProvider
from coord_harness.models.base import ModelClient
from coord_harness.models.mock import MockModelClient
from coord_harness.models.openrouter import OpenRouterClient


def build_model_client(model_spec: ModelSpec) -> ModelClient:
    if model_spec.provider is ModelProvider.MOCK:
        return MockModelClient(model_spec)
    if model_spec.provider is ModelProvider.OPENROUTER:
        return OpenRouterClient(model_spec)
    raise ValueError(f"Unsupported model provider: {model_spec.provider.value}")
