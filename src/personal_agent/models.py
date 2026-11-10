"""Shared task states, tool interfaces and finite execution limits."""
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol
import hashlib
import json
import math
import time


def canonical(value: Any) -> str:
    """Encode protocol data deterministically, rejecting non-finite numbers."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def decode(value: str | bytes) -> Any:
    """Reject duplicate object keys and non-finite numbers at JSON input."""
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    def invalid(value):
        raise ValueError("non-finite JSON number")

    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)


class AgentError(Exception):
    """Safe public error code; internal exceptions are never model output."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class TaskStatus(StrEnum):
    READY = "ready"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    WAITING_CONFIRMATION = "waiting_confirmation"
    WAITING_EVENT = "waiting_event"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BLOCKED = "blocked"


TERMINAL = frozenset((TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED))


class ActionStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    UNKNOWN = "unknown"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ReceiptStatus(StrEnum):
    COMMITTED = "committed"
    NOT_COMMITTED = "not_committed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Receipt:
    status: ReceiptStatus
    result: Any = None


@dataclass(frozen=True)
class Budget:
    calls: int = 16
    generations: int = 24
    output_tokens: int = 8192
    context_bytes: int = 65536
    output_bytes: int = 32768
    wall_seconds: float = 60
    lease_seconds: float = 90
    confirmation_seconds: float = 900

    def __post_init__(self):
        for field in ("calls", "generations", "output_tokens", "context_bytes", "output_bytes"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("wall_seconds", "lease_seconds", "confirmation_seconds"):
            value = getattr(self, field)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be positive and finite")
        if self.lease_seconds <= self.wall_seconds:
            raise ValueError("lease_seconds must exceed wall_seconds")


@dataclass(frozen=True)
class Turn:
    content: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    adapter: str
    remote_name: str
    readonly: bool = True

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "inputSchema": self.schema}


class Policy(Protocol):
    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 remaining: dict[str, Any]) -> Turn: ...


class Adapter(Protocol):
    fingerprint: str
    idempotent: bool

    async def list_tools(self) -> list[Tool]: ...
    async def call(self, name: str, arguments: dict[str, Any],
                   operation_id: str | None) -> Any: ...
    async def reconcile(self, operation_id: str) -> Receipt: ...


def utc_now() -> float:
    return time.time()
