from __future__ import annotations

import math
from dataclasses import dataclass, field


class BudgetExceededError(RuntimeError):
    """Raised when a trial exceeds the configured billed token budget."""


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def truncate_to_token_limit(text: str, token_limit: int) -> str:
    if token_limit <= 0:
        return ""
    if estimate_text_tokens(text) <= token_limit:
        return text
    char_limit = token_limit * 4
    if char_limit <= 1:
        return text[:char_limit]
    return text[: max(char_limit - 1, 0)].rstrip() + "…"


@dataclass
class BudgetLedger:
    total_billed_token_budget: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    events: list[dict[str, int | str]] = field(default_factory=list)

    def consume(self, *, prompt_tokens: int, completion_tokens: int, label: str) -> None:
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token usage cannot be negative.")
        proposed_total = self.prompt_tokens + self.completion_tokens + prompt_tokens + completion_tokens
        if proposed_total > self.total_billed_token_budget:
            raise BudgetExceededError(
                f"Budget exceeded for {label}: {proposed_total} > {self.total_billed_token_budget}"
            )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.events.append(
            {
                "label": label,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "billed_tokens": prompt_tokens + completion_tokens,
            }
        )

    @property
    def billed_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def remaining_tokens(self) -> int:
        return self.total_billed_token_budget - self.billed_tokens

    def snapshot(self) -> dict[str, int]:
        return {
            "total_billed_token_budget": self.total_billed_token_budget,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "billed_tokens": self.billed_tokens,
            "remaining_tokens": self.remaining_tokens,
        }
