from __future__ import annotations

import pytest

from coord_harness.core.budget import BudgetExceededError, BudgetLedger, truncate_to_token_limit


def test_budget_ledger_tracks_prompt_and_completion() -> None:
    ledger = BudgetLedger(total_billed_token_budget=20)
    ledger.consume(prompt_tokens=5, completion_tokens=4, label="call-1")
    assert ledger.billed_tokens == 9
    assert ledger.remaining_tokens == 11
    assert ledger.snapshot()["prompt_tokens"] == 5


def test_budget_ledger_raises_on_overflow() -> None:
    ledger = BudgetLedger(total_billed_token_budget=10)
    ledger.consume(prompt_tokens=4, completion_tokens=4, label="call-1")
    with pytest.raises(BudgetExceededError):
        ledger.consume(prompt_tokens=2, completion_tokens=1, label="call-2")


def test_message_truncation_respects_zero_budget() -> None:
    assert truncate_to_token_limit("hello world", 0) == ""
