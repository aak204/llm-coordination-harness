from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from coord_harness.config.models import ModelSpec
from coord_harness.core.budget import estimate_text_tokens
from coord_harness.core.protocols import render_decision_prompt, sanitize_public_rationale, strip_reasoning_blocks
from coord_harness.core.types import BenchmarkTask, GenerationResult
from coord_harness.models.base import ModelClient


ANSWER_RE = re.compile(r"^ANSWER:\s*(?P<answer>[A-Za-z0-9_-]+)\s*$", re.MULTILINE)
CONFIDENCE_RE = re.compile(r"^CONFIDENCE:\s*(?P<confidence>[0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
RATIONALE_RE = re.compile(r"^RATIONALE:\s*(?P<rationale>.+)$", re.MULTILINE)

_CATALOG_CACHE_LOCK = threading.Lock()
_CATALOG_CACHE: dict[str, Any] = {}
_ENDPOINTS_CACHE_LOCK = threading.Lock()
_ENDPOINTS_CACHE: dict[str, dict[str, Any]] = {}


class OpenRouterClient(ModelClient):
    def __init__(self, model_spec: ModelSpec, timeout_seconds: float = 60.0):
        self.model_spec = model_spec
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.coord-harness",
            "X-Title": "coord_harness",
        }
        headers.update(model_spec.headers)
        self._client = httpx.Client(
            base_url="https://openrouter.ai/api/v1",
            headers=headers,
            timeout=timeout_seconds,
        )
        self._catalog_snapshot: dict | None = None
        self._catalog_snapshot_hash: str | None = None
        self._catalog_snapshot_fetched_at: str | None = None
        self._model_catalog_entry: dict | None = None
        self._endpoints_snapshot: dict | None = None
        self._endpoints_snapshot_hash: str | None = None
        self._endpoints_snapshot_fetched_at: str | None = None
        self._max_retries = 8

    def _provider_payload(self) -> dict[str, Any]:
        routing = self.model_spec.routing
        payload: dict[str, Any] = {
            "allow_fallbacks": self.model_spec.allow_auto_routing or routing.allow_fallbacks,
        }
        if routing.order:
            payload["order"] = routing.order
        if routing.only:
            payload["only"] = routing.only
        if routing.ignore:
            payload["ignore"] = routing.ignore
        if routing.require_parameters:
            payload["require_parameters"] = True
        return payload

    @staticmethod
    def _content_text(content: Any) -> str | None:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            joined = "".join(parts).strip()
            return joined or None
        return None

    def _reasoning_payload(self, *, enable_reasoning: bool) -> dict[str, Any] | None:
        reasoning = self.model_spec.reasoning
        payload: dict[str, Any] = {}
        if reasoning.enabled is not None:
            payload["enabled"] = reasoning.enabled
        if reasoning.effort is not None:
            payload["effort"] = reasoning.effort
        if reasoning.max_tokens is not None:
            payload["max_tokens"] = reasoning.max_tokens
        if reasoning.exclude:
            payload["exclude"] = True
        if not enable_reasoning and payload.get("enabled") is True:
            payload["enabled"] = False
        return payload or None

    def _snapshot_hash(self, payload: dict | list) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _ensure_catalog_snapshot(self) -> None:
        if self._catalog_snapshot is not None:
            return
        with _CATALOG_CACHE_LOCK:
            cached_snapshot = _CATALOG_CACHE.get("snapshot")
            if cached_snapshot is not None:
                self._catalog_snapshot = cached_snapshot
                self._catalog_snapshot_hash = _CATALOG_CACHE.get("snapshot_hash")
                self._catalog_snapshot_fetched_at = _CATALOG_CACHE.get("snapshot_fetched_at")
                self._model_catalog_entry = _CATALOG_CACHE.get("entries", {}).get(self.model_spec.model_id)
                return
        try:
            response = self._client.get("/models")
            response.raise_for_status()
        except httpx.HTTPError:
            self._catalog_snapshot = {"data": [], "snapshot_error": "catalog_fetch_failed"}
            self._catalog_snapshot_hash = self._snapshot_hash(self._catalog_snapshot)
            self._catalog_snapshot_fetched_at = datetime.now(UTC).isoformat()
            self._model_catalog_entry = None
            with _CATALOG_CACHE_LOCK:
                _CATALOG_CACHE["snapshot"] = self._catalog_snapshot
                _CATALOG_CACHE["snapshot_hash"] = self._catalog_snapshot_hash
                _CATALOG_CACHE["snapshot_fetched_at"] = self._catalog_snapshot_fetched_at
                _CATALOG_CACHE["entries"] = {}
            return
        self._catalog_snapshot = response.json()
        self._catalog_snapshot_hash = self._snapshot_hash(self._catalog_snapshot)
        self._catalog_snapshot_fetched_at = datetime.now(UTC).isoformat()
        entry_map = {}
        for entry in self._catalog_snapshot.get("data", []):
            entry_map[entry.get("id")] = entry
            if entry.get("id") == self.model_spec.model_id:
                self._model_catalog_entry = entry
        with _CATALOG_CACHE_LOCK:
            _CATALOG_CACHE["snapshot"] = self._catalog_snapshot
            _CATALOG_CACHE["snapshot_hash"] = self._catalog_snapshot_hash
            _CATALOG_CACHE["snapshot_fetched_at"] = self._catalog_snapshot_fetched_at
            _CATALOG_CACHE["entries"] = entry_map

    def _ensure_endpoints_snapshot(self) -> None:
        if self._endpoints_snapshot is not None:
            return
        author, slug = self.model_spec.model_id.split("/", 1)
        with _ENDPOINTS_CACHE_LOCK:
            cached = _ENDPOINTS_CACHE.get(self.model_spec.model_id)
            if cached is not None:
                self._endpoints_snapshot = cached["snapshot"]
                self._endpoints_snapshot_hash = cached["snapshot_hash"]
                self._endpoints_snapshot_fetched_at = cached["snapshot_fetched_at"]
                return
        try:
            response = self._client.get(f"/models/{author}/{slug}/endpoints")
            response.raise_for_status()
            payload = response.json()
            self._endpoints_snapshot = payload.get("data", {})
        except httpx.HTTPError:
            self._endpoints_snapshot = {"endpoints": [], "snapshot_error": "endpoints_fetch_failed"}
        self._endpoints_snapshot_hash = self._snapshot_hash(self._endpoints_snapshot)
        self._endpoints_snapshot_fetched_at = datetime.now(UTC).isoformat()
        with _ENDPOINTS_CACHE_LOCK:
            _ENDPOINTS_CACHE[self.model_spec.model_id] = {
                "snapshot": self._endpoints_snapshot,
                "snapshot_hash": self._endpoints_snapshot_hash,
                "snapshot_fetched_at": self._endpoints_snapshot_fetched_at,
            }

    def _resolve_provider_endpoint(self, resolved_provider_name: str | None) -> dict[str, Any] | None:
        if not resolved_provider_name:
            return None
        self._ensure_endpoints_snapshot()
        endpoints = self._endpoints_snapshot.get("endpoints", []) if self._endpoints_snapshot else []
        for endpoint in endpoints:
            if endpoint.get("provider_name") == resolved_provider_name:
                return endpoint
        return None

    def _fallback_status(self, resolved_provider_tag: str | None) -> tuple[bool | None, str]:
        requested_policy = self._provider_payload()
        requested_priority = requested_policy.get("order") or requested_policy.get("only") or []
        if not requested_priority:
            return None, "unconstrained_route_policy"
        if resolved_provider_tag is None:
            return None, "resolved_provider_tag_unavailable"
        if resolved_provider_tag == requested_priority[0]:
            return False, "matched_primary_requested_provider"
        if resolved_provider_tag in requested_priority[1:]:
            return True, "resolved_to_non_primary_requested_provider"
        return True, "resolved_outside_requested_provider_policy"

    def generate_decision(
        self,
        *,
        task: BenchmarkTask,
        agent_id: str,
        visible_messages: list[str],
        enable_reasoning: bool,
        seed: int,
    ) -> GenerationResult:
        self._ensure_catalog_snapshot()
        self._ensure_endpoints_snapshot()
        prompt = render_decision_prompt(
            task=task,
            agent_id=agent_id,
            visible_messages=visible_messages,
            enable_reasoning=enable_reasoning,
        )
        request_payload = {
            "model": self.model_spec.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.model_spec.temperature,
            "top_p": self.model_spec.top_p,
            "max_tokens": self.model_spec.max_completion_tokens,
            "seed": seed,
            "provider": self._provider_payload(),
            "stream": False,
            "usage": {"include": True},
        }
        reasoning_payload = self._reasoning_payload(enable_reasoning=enable_reasoning)
        if reasoning_payload is not None:
            request_payload["reasoning"] = reasoning_payload
        response_payload: dict[str, Any] | None = None
        content: str | None = None
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=request_payload)
                response.raise_for_status()
                response_payload = response.json()
                if "error" in response_payload:
                    raise ValueError(f"OpenRouter returned error payload: {response_payload['error']}")
                choices = response_payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ValueError("OpenRouter returned payload without choices.")
                message = choices[0].get("message")
                if not isinstance(message, dict):
                    raise ValueError("OpenRouter returned payload without a message object.")
                content = self._content_text(message.get("content"))
                if content is None:
                    raise ValueError("OpenRouter returned null content.")
                break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
                last_error = exc
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status = exc.response.status_code
                if status not in {408, 429, 500, 502, 503, 504}:
                    raise
            except ValueError as exc:
                last_error = exc
            if attempt < self._max_retries:
                retry_after = None
                if isinstance(last_error, httpx.HTTPStatusError):
                    retry_after_header = last_error.response.headers.get("Retry-After")
                    if retry_after_header and retry_after_header.isdigit():
                        retry_after = int(retry_after_header)
                sleep_seconds = retry_after if retry_after is not None else min(2 ** attempt, 30)
                time.sleep(sleep_seconds)
        if response_payload is None or content is None:
            if last_error:
                raise last_error
            raise RuntimeError("OpenRouter completion failed without a captured error.")
        public_content = strip_reasoning_blocks(content)
        answer_match = ANSWER_RE.search(public_content)
        confidence_match = CONFIDENCE_RE.search(public_content)
        rationale_match = RATIONALE_RE.search(public_content)

        resolved_provider_name = response_payload.get("provider")
        endpoint = self._resolve_provider_endpoint(resolved_provider_name)
        resolved_provider_tag = endpoint.get("tag") if endpoint else None
        fallback_used, fallback_basis = self._fallback_status(resolved_provider_tag)
        usage = response_payload.get("usage", {})

        return GenerationResult(
            answer=answer_match.group("answer") if answer_match else task.answer_choices[0],
            confidence=round(
                max(0.0, min(float(confidence_match.group("confidence")), 1.0)) if confidence_match else 0.5,
                2,
            ),
            rationale=sanitize_public_rationale(rationale_match.group("rationale")) if rationale_match else "No rationale returned.",
            raw_text=content,
            prompt_tokens=int(usage.get("prompt_tokens", estimate_text_tokens(prompt))),
            completion_tokens=int(usage.get("completion_tokens", estimate_text_tokens(content))),
            route_metadata={
                "request_id": response_payload.get("id"),
                "requested_model_id": self.model_spec.model_id,
                "requested_provider_policy": self._provider_payload(),
                "requested_reasoning_policy": reasoning_payload,
                "resolved_model_id": response_payload.get("model", self.model_spec.model_id),
                "resolved_provider_name": resolved_provider_name,
                "resolved_provider_tag": resolved_provider_tag,
                "fallback_used": fallback_used,
                "fallback_inference_basis": fallback_basis,
                "service_tier": response_payload.get("service_tier"),
                "pricing_snapshot": {
                    "catalog_pricing": self._model_catalog_entry.get("pricing") if self._model_catalog_entry else None,
                    "endpoint_pricing": endpoint.get("pricing") if endpoint else None,
                },
                "usage_snapshot": usage,
                "catalog_snapshot_hash": self._catalog_snapshot_hash,
                "catalog_snapshot_fetched_at": self._catalog_snapshot_fetched_at,
                "endpoints_snapshot_hash": self._endpoints_snapshot_hash,
                "endpoints_snapshot_fetched_at": self._endpoints_snapshot_fetched_at,
            },
        )

    def fetch_model_catalog_snapshot(self) -> dict | None:
        self._ensure_catalog_snapshot()
        return self._catalog_snapshot

    def describe_runtime_metadata(self) -> dict[str, Any]:
        self._ensure_catalog_snapshot()
        self._ensure_endpoints_snapshot()
        return {
            "requested_model_id": self.model_spec.model_id,
            "requested_provider_policy": self._provider_payload(),
            "requested_reasoning_policy": self._reasoning_payload(enable_reasoning=False),
            "catalog_pricing": self._model_catalog_entry.get("pricing") if self._model_catalog_entry else None,
            "catalog_created": self._model_catalog_entry.get("created") if self._model_catalog_entry else None,
            "catalog_snapshot_hash": self._catalog_snapshot_hash,
            "catalog_snapshot_fetched_at": self._catalog_snapshot_fetched_at,
            "endpoints_snapshot_hash": self._endpoints_snapshot_hash,
            "endpoints_snapshot_fetched_at": self._endpoints_snapshot_fetched_at,
            "endpoint_count": len(self._endpoints_snapshot.get("endpoints", [])) if self._endpoints_snapshot else 0,
        }

    def close(self) -> None:
        self._client.close()
