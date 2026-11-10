"""Sequential conversation execution with durable confirmations and reconciliation."""
import asyncio
import time
from .context import build_context
from .models import (AgentError, ActionStatus, Budget, ReceiptStatus, TaskStatus,
                     canonical, digest, utc_now)
from .store import identity
from .tools import Registry, builtin


def failure(code: str) -> dict:
    return {"ok": False, "error": {"code": code}}


class Runtime:
    def __init__(self, store, policy, registry: Registry | None = None,
                 budget: Budget | None = None, clock=utc_now):
        self.store, self.policy = store, policy
        self.registry = registry or Registry()
        self.budget, self.clock = budget or Budget(), clock

    def bounded(self, result):
        if len(canonical(result).encode()) > self.budget.output_bytes:
            return failure("OUTPUT_TOO_LARGE")
        return result

    async def _action(self, action, tid, token, timeout):
        """Resolve an existing journal entry before requesting another model turn."""
        deadline = time.monotonic() + timeout
        aid, name = action["id"], action["tool"]
        expected = digest([self.store.user_id, tid, name, action["arguments"],
                           action["fingerprint"], action["expires_at"]])
        if expected != action["request_sha256"]:
            self.store.transition(tid, token, TaskStatus.BLOCKED, self.clock(), "CONFIRMATION_REQUEST_CHANGED")
            return False
        tool = self.registry.tools.get(name)
        if not tool or self.registry.bindings.get(name) != action["fingerprint"]:
            if action["status"] in (ActionStatus.EXECUTING, ActionStatus.UNKNOWN):
                self.store.transition(tid, token, TaskStatus.BLOCKED, self.clock(), "ADAPTER_CHANGED")
                return False
            self.store.settle(aid, tid, token, failure("ADAPTER_CHANGED"), self.clock())
            return True
        adapter = self.registry.adapters[tool.adapter]
        if action["status"] == ActionStatus.APPROVED:
            if self.clock() >= action["expires_at"]:
                self.store.settle(aid, tid, token, failure("CONFIRMATION_EXPIRED"), self.clock())
                return True
            retry = True
        elif action["status"] in (ActionStatus.EXECUTING, ActionStatus.UNKNOWN):
            self.store.action_status(aid, tid, token, ActionStatus.UNKNOWN, self.clock())
            try:
                receipt = await asyncio.wait_for(adapter.reconcile(aid), max(0, deadline - time.monotonic()))
            except Exception:
                receipt = None
            if receipt and receipt.status == ReceiptStatus.COMMITTED:
                self.store.settle(aid, tid, token, self.bounded(receipt.result), self.clock())
                return True
            retry = bool(receipt and receipt.status == ReceiptStatus.NOT_COMMITTED and adapter.idempotent)
            if not retry:
                self.store.transition(tid, token, TaskStatus.BLOCKED, self.clock(), "OPERATION_OUTCOME_UNKNOWN")
                return False
            if self.clock() >= action["expires_at"]:
                self.store.settle(aid, tid, token, failure("CONFIRMATION_EXPIRED"), self.clock())
                return True
        else:
            self.store.transition(tid, token, TaskStatus.WAITING_CONFIRMATION, self.clock())
            return False
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            self.store.transition(tid, token, TaskStatus.PAUSED, self.clock(), "RUN_BUDGET_EXHAUSTED")
            return False
        self.store.action_status(aid, tid, token, ActionStatus.EXECUTING, self.clock())
        try:
            result = await asyncio.wait_for(adapter.call(tool.remote_name, action["arguments"], aid), timeout)
            result = self.bounded(result)
        except Exception:
            self.store.action_status(aid, tid, token, ActionStatus.UNKNOWN, self.clock())
            self.store.transition(tid, token, TaskStatus.BLOCKED, self.clock(), "OPERATION_OUTCOME_UNKNOWN")
            return False
        self.store.settle(aid, tid, token, result, self.clock())
        return True

    async def run(self, tid: str) -> dict:
        token = self.store.claim(tid, self.clock(), self.budget.lease_seconds)
        if token is None:
            return self.store.task(tid)
        start = time.monotonic()
        calls = generations = outputs = 0

        def remaining():
            return self.budget.wall_seconds - (time.monotonic() - start)

        try:
            action = self.store.active_action(tid)
            if action:
                if not await self._action(action, tid, token, remaining()):
                    return self.store.task(tid)
                calls += 1
            while True:
                now = self.clock()
                task = self.store.guard(tid, token, now)
                if (calls >= self.budget.calls or generations >= self.budget.generations
                        or outputs >= self.budget.output_tokens or remaining() <= 0):
                    self.store.transition(tid, token, TaskStatus.PAUSED, now, "RUN_BUDGET_EXHAUSTED")
                    break
                tools = self.registry.public()
                messages = build_context(self.store, task, tools, self.budget.context_bytes, now)
                limits = {"generated_tokens": self.budget.output_tokens - outputs,
                          "wall_seconds": remaining()}
                turn = await asyncio.wait_for(asyncio.to_thread(self.policy.generate, messages, tools, limits), remaining())
                self.store.guard(tid, token, self.clock())
                if remaining() <= 0:
                    self.store.transition(tid, token, TaskStatus.PAUSED, self.clock(), "RUN_BUDGET_EXHAUSTED")
                    break
                generations += 1
                if (type(turn.output_tokens) is not int or turn.output_tokens < 0
                        or type(turn.input_tokens) is not int or turn.input_tokens < 0):
                    raise AgentError("INVALID_MODEL_RESPONSE")
                outputs += turn.output_tokens
                if outputs > self.budget.output_tokens:
                    raise AgentError("MODEL_OUTPUT_BUDGET_EXCEEDED")
                if len(canonical([turn.content, turn.tool, turn.arguments]).encode()) > self.budget.output_bytes:
                    raise AgentError("MODEL_OUTPUT_TOO_LARGE")
                if turn.tool is None:
                    if not isinstance(turn.content, str) or not turn.content.strip():
                        raise AgentError("INVALID_MODEL_RESPONSE")
                    self.store.transition(tid, token, TaskStatus.COMPLETED, self.clock(),
                                          payload=[{"role": "assistant", "content": turn.content}])
                    break
                calls += 1
                call = {"role": "assistant", "content": turn.content,
                        "tool_call": {"id": identity("call"), "name": turn.tool, "arguments": turn.arguments}}
                try:
                    tool = self.registry.validate(turn.tool, turn.arguments)
                except AgentError as exc:
                    self._exchange(tid, token, call, failure(exc.code))
                    continue
                if tool.adapter and not tool.readonly:
                    self.store.prepare_action(tid, token, tool.name, turn.arguments,
                                              self.registry.bindings[tool.name], call,
                                              self.clock(), self.budget.confirmation_seconds)
                    break
                if not tool.adapter:
                    with self.store.transaction():
                        task = self.store.guard(tid, token, self.clock())
                        self.store.db.execute("SAVEPOINT builtin")
                        try:
                            data, status, wait = builtin(self.store, tid, tool.name, turn.arguments, self.clock())
                            result = self.bounded({"ok": True, "data": data})
                            self.store.db.execute("RELEASE builtin")
                        except AgentError as exc:
                            self.store.db.execute("ROLLBACK TO builtin")
                            self.store.db.execute("RELEASE builtin")
                            result, status, wait = failure(exc.code), None, {}
                        payload = [call, {"role": "tool", "tool_call_id": call["tool_call"]["id"], "content": canonical(result)}]
                        if status in (TaskStatus.WAITING_USER, TaskStatus.COMPLETED):
                            text = turn.arguments["question" if status == TaskStatus.WAITING_USER else "answer"]
                            payload.append({"role": "assistant", "content": text})
                        self.store.guard(tid, token, self.clock())
                        self.store._message(task, payload, self.clock())
                        if status:
                            self.store._state(tid, status, self.clock(), **wait)
                    if status:
                        break
                else:
                    try:
                        result = await asyncio.wait_for(
                            self.registry.adapters[tool.adapter].call(tool.remote_name, turn.arguments, None), remaining())
                        result = self.bounded(result)
                    except Exception:
                        result = failure("APPLICATION_READ_FAILED")
                    self._exchange(tid, token, call, result)
        except AgentError as exc:
            if exc.code != "STALE_LEASE":
                self._stop(tid, token, TaskStatus.PAUSED if exc.code == "CONTEXT_BUDGET_EXHAUSTED" else TaskStatus.FAILED, exc.code)
        except TimeoutError:
            self._stop(tid, token, TaskStatus.PAUSED, "RUN_BUDGET_EXHAUSTED")
        except Exception:
            self._stop(tid, token, TaskStatus.FAILED, "RUNTIME_ERROR")
        return self.store.task(tid)

    def _stop(self, tid, token, status, code):
        try:
            self.store.transition(tid, token, status, self.clock(), code)
        except AgentError as exc:
            if exc.code != "STALE_LEASE":
                raise

    def _exchange(self, tid, token, call, result):
        self.store.append(tid, token, [call, {"role": "tool", "tool_call_id": call["tool_call"]["id"],
                                           "content": canonical(result)}], self.clock())

    async def tick(self, limit: int = 10) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise AgentError("INVALID_LIMIT")
        results = []
        for tid in self.store.ready(self.clock(), limit):
            results.append(await self.run(tid))
        return results
