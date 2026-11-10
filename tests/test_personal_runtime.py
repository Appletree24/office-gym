"""Conversation and recovery tests exercise the real store and bounded runtime."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from personal_agent.cli import ScriptPolicy, demo
from personal_agent.context import build_context
from personal_agent.models import (ActionStatus, AgentError, Budget, Receipt, ReceiptStatus,
                                   TaskStatus, Tool, Turn, canonical)
from personal_agent.runtime import Runtime
from personal_agent.store import Store
from personal_agent.tools import Registry, obj, text
from officegym.harness.policy import render_messages


class FakeApplication:
    fingerprint = "fixture-app-v1"
    idempotent = True

    def __init__(self):
        self.calls = []
        self.effects = {}
        self.fail_after_write = False
        self.receipt_status = None

    async def list_tools(self):
        return [Tool("send", "Send to a controlled fixture.", obj({"body": text()}), "", "send", False)]

    async def call(self, name, args, operation_id):
        self.calls.append((args, operation_id))
        self.effects.setdefault(operation_id, {"ok": True, "data": {"sent": args["body"]}})
        if self.fail_after_write:
            raise ConnectionError("not exposed to the model")
        return self.effects[operation_id]

    async def reconcile(self, operation_id):
        if self.receipt_status is not None:
            return Receipt(self.receipt_status)
        return Receipt(ReceiptStatus.COMMITTED, self.effects[operation_id]) if operation_id in self.effects else Receipt(ReceiptStatus.NOT_COMMITTED)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "state.sqlite"
        self.store = Store(self.path, "alice")
        self.now = 100
        self.app = FakeApplication()
        self.registry = Registry()
        await self.registry.add("app", self.app)
        self.task = self.store.submit("help", self.now)

    async def asyncTearDown(self):
        self.store.close()
        self.temp.cleanup()

    def runtime(self, turns, budget=None):
        return Runtime(self.store, ScriptPolicy(turns), self.registry, budget, lambda: self.now)

    async def prepare(self):
        result = await self.runtime([Turn(tool="app__send", arguments={"body": "hello"})]).run(self.task["id"])
        self.assertEqual(result["status"], "waiting_confirmation")
        self.assertEqual(self.app.calls, [])
        return self.store.active_action(self.task["id"])

    async def test_plain_text_completes_one_task_and_session_remains_usable(self):
        other = self.store.submit("another task", self.now, self.task["session_id"])
        result = await self.runtime([Turn(content="hello")]).run(self.task["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.store.task(other["id"])["status"], "ready")
        self.assertIsNotNone(self.store.submit("next", self.now, self.task["session_id"]))

    async def test_clarification_resume_and_cross_session_memory(self):
        tid = self.task["id"]
        self.assertEqual((await self.runtime([Turn(tool="task_ask", arguments={"question": "name?"})]).run(tid))["status"], "waiting_user")
        self.store.submit("Alice", self.now, task_id=tid)
        source = self.store.history(self.task["session_id"])[-1]["id"]
        await self.runtime([Turn(tool="memory_put", arguments={"key": "name", "kind": "preference", "content": "Alice", "source_ids": [source], "expected_version": 0}), Turn(content="saved")]).run(tid)
        other = self.store.submit("remember me?", self.now)
        seen = []
        def answer(messages):
            seen.extend(messages)
            return Turn(content="Hello Alice")
        await self.runtime([answer]).run(other["id"])
        self.assertIn("Alice", canonical(seen))
        self.assertNotEqual(other["session_id"], self.task["session_id"])

    async def test_wait_and_independent_scheduled_task_survive_reopen(self):
        tid = self.task["id"]
        await self.runtime([Turn(tool="task_schedule", arguments={"instruction": "remind", "until": 110}),
                            Turn(tool="task_wait", arguments={"until": 120})]).run(tid)
        self.store.close()
        self.store = Store(self.path, "alice")
        self.now = 110
        results = await self.runtime([Turn(content="reminder")]).tick()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(self.store.task(tid)["status"], "waiting_event")
        self.now = 120
        self.assertEqual((await self.runtime([Turn(content="awake")]).tick())[0]["id"], tid)

    async def test_budget_pauses_and_resumes_without_ending_session(self):
        runtime = self.runtime([Turn(tool="task_list", arguments={})], Budget(calls=1))
        result = await runtime.run(self.task["id"])
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["reason"], "RUN_BUDGET_EXHAUSTED")
        self.store.resume(self.task["id"], self.now)
        self.assertEqual((await self.runtime([Turn(content="done")]).run(self.task["id"]))["status"], "completed")

    async def test_approval_and_rejection_are_not_model_tools(self):
        self.assertNotIn("approve", self.registry.tools)
        action = await self.prepare()
        bob = Store(self.path, "bob")
        try:
            with self.assertRaisesRegex(AgentError, "NOT_FOUND"):
                bob.decide(action["id"], True, self.now)
        finally:
            bob.close()
        self.store.decide(action["id"], False, self.now)
        await self.runtime([Turn(content="rejected")]).run(self.task["id"])
        self.assertEqual(self.app.calls, [])
        self.assertEqual(self.store.action(action["id"])["status"], "rejected")

    async def test_approved_write_only_executes_once_and_history_pairs_are_complete(self):
        action = await self.prepare()
        self.store.decide(action["id"], True, self.now)
        result = await self.runtime([Turn(content="sent")]).run(self.task["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.app.calls), 1)
        self.assertEqual(self.app.calls[0][1], action["id"])
        self.assertEqual(self.store.action(action["id"])["status"], "succeeded")
        history = self.store.history(self.task["session_id"])
        exchange = history[-2]["payload"]
        self.assertEqual(exchange[0]["tool_call"]["id"], exchange[1]["tool_call_id"])

    async def test_expiry_and_route_change_prevent_execution(self):
        action = await self.prepare()
        self.store.decide(action["id"], True, self.now)
        self.now = action["expires_at"]
        await self.runtime([Turn(content="expired")]).run(self.task["id"])
        self.assertEqual(self.app.calls, [])
        self.assertEqual(self.store.action(action["id"])["result"]["error"]["code"], "CONFIRMATION_EXPIRED")
        self.task = self.store.submit("new task", self.now)
        action = await self.prepare()
        self.store.decide(action["id"], True, self.now)
        self.registry.bindings["app__send"] = "new-account"
        await self.runtime([Turn(content="changed")]).run(self.task["id"])
        self.assertEqual(self.app.calls, [])
        self.assertEqual(self.store.action(action["id"])["result"]["error"]["code"], "ADAPTER_CHANGED")

    async def test_changed_confirmation_arguments_are_blocked(self):
        action = await self.prepare()
        self.store.decide(action["id"], True, self.now)
        self.store.db.execute("UPDATE actions SET arguments=? WHERE id=?", (canonical({"body": "changed"}), action["id"]))
        result = await self.runtime([]).run(self.task["id"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "CONFIRMATION_REQUEST_CHANGED")
        self.assertEqual(self.app.calls, [])

    async def test_new_user_input_invalidates_an_unexecuted_approval(self):
        action = await self.prepare()
        self.store.decide(action["id"], True, self.now)
        self.store.submit("Do not send that; just answer me.", self.now, task_id=self.task["id"])
        result = await self.runtime([Turn(content="Understood")]).run(self.task["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.store.action(action["id"])["status"], "rejected")
        self.assertEqual(self.app.calls, [])

    async def test_write_then_disconnect_reconciles_after_restart_without_duplicate(self):
        action = await self.prepare()
        self.store.decide(action["id"], True, self.now)
        self.app.fail_after_write = True
        result = await self.runtime([]).run(self.task["id"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.store.action(action["id"])["status"], "unknown")
        self.store.close()
        self.store = Store(self.path, "alice")
        self.store.resume(self.task["id"], self.now)
        result = await self.runtime([Turn(content="receipt recovered")]).run(self.task["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.app.calls), 1)
        self.assertEqual(len(self.app.effects), 1)

    async def test_unknown_receipt_is_blocked_and_known_absence_can_retry_same_key(self):
        action = await self.prepare()
        self.store.decide(action["id"], True, self.now)
        token = self.store.claim(self.task["id"], self.now, 10)
        self.store.action_status(action["id"], self.task["id"], token, ActionStatus.EXECUTING, self.now)
        self.now += 10
        self.app.receipt_status = ReceiptStatus.UNKNOWN
        result = await self.runtime([]).run(self.task["id"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.app.calls, [])
        self.store.resume(self.task["id"], self.now)
        self.app.receipt_status = ReceiptStatus.NOT_COMMITTED
        self.app.idempotent = False
        result = await self.runtime([]).run(self.task["id"])
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(self.app.calls, [])
        self.store.resume(self.task["id"], self.now)
        self.app.idempotent = True
        await self.runtime([Turn(content="recovered")]).run(self.task["id"])
        self.assertEqual(self.app.calls[0][1], action["id"])

    async def test_cancellation_discards_late_model_and_application_results(self):
        tid = self.task["id"]
        def cancelled(messages):
            # Simulate another process cancelling while generation is in flight.
            other = Store(self.path, "alice")
            try:
                other.cancel(tid, self.now)
            finally:
                other.close()
            return Turn(content="late")
        result = await self.runtime([cancelled]).run(tid)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(len(self.store.history(self.task["session_id"])), 1)
        self.task = self.store.submit("send", self.now)
        action = await self.prepare()
        self.store.decide(action["id"], True, self.now)
        original = self.app.call
        async def late(name, args, operation_id):
            result = await original(name, args, operation_id)
            self.store.cancel(self.task["id"], self.now)
            return result
        self.app.call = late
        result = await self.runtime([]).run(self.task["id"])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(self.store.action(action["id"])["status"], "unknown")

    async def test_context_preserves_tool_pairs_and_reports_omissions(self):
        tid = self.task["id"]
        token = self.store.claim(tid, self.now, 100)
        for i in range(30):
            self.store.append(tid, token, [{"role": "assistant", "tool_call": {"id": str(i), "name": "test", "arguments": {}}},
                                          {"role": "tool", "tool_call_id": str(i), "content": "x" * 1000}], self.now)
        messages = build_context(self.store, self.store.task(tid), [], 5000, self.now)
        self.assertLessEqual(len(canonical({"messages": render_messages(messages), "tools": []}).encode()), 5000)
        meta = json.loads(messages[1]["content"].removeprefix("Runtime context: "))
        self.assertGreater(meta["omitted_history_groups"], 0)
        for index, message in enumerate(messages):
            if "tool_call" in message:
                self.assertEqual(messages[index + 1]["tool_call_id"], message["tool_call"]["id"])

    async def test_pending_confirmation_expires_in_scheduler(self):
        action = await self.prepare()
        self.now = action["expires_at"]
        results = await self.runtime([Turn(content="expired")]).tick()
        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(self.store.action(action["id"])["status"], "expired")
        self.assertEqual(self.app.calls, [])

    async def test_expired_inflight_write_is_recovered_by_receipt_without_resending(self):
        action = await self.prepare()
        self.store.decide(action["id"], True, self.now)
        original = self.app.call
        async def delayed(name, args, operation_id):
            result = await original(name, args, operation_id)
            self.now += 91
            return result
        self.app.call = delayed
        result = await self.runtime([]).run(self.task["id"])
        self.assertEqual(result["status"], "running")
        self.assertEqual(self.store.action(action["id"])["status"], "executing")
        result = await self.runtime([Turn(content="reconciled")]).run(self.task["id"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.app.calls), 1)

    async def test_concurrent_runtime_does_not_generate_a_second_turn(self):
        import threading
        started, finish = threading.Event(), threading.Event()
        def generate(messages):
            started.set()
            if not finish.wait(2):
                raise RuntimeError("test did not release generation")
            return Turn(content="one answer")
        pending = asyncio.create_task(self.runtime([generate]).run(self.task["id"]))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            result = await self.runtime([]).run(self.task["id"])
            self.assertEqual(result["status"], "running")
        finally:
            finish.set()
            result = await asyncio.wait_for(pending, 3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(self.store.history(self.task["session_id"])), 2)

    async def test_context_overflow_and_invalid_tool_are_bounded_failures(self):
        result = await self.runtime([], Budget(context_bytes=10)).run(self.task["id"])
        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["reason"], "CONTEXT_BUDGET_EXHAUSTED")
        self.store.resume(self.task["id"], self.now)
        result = await self.runtime([Turn(tool="memory_put", arguments={"principal_id": "bob"})],
                                    Budget(calls=1)).run(self.task["id"])
        self.assertEqual(result["status"], "paused")
        self.assertEqual(self.store.memory_search(), [])
        self.assertIn("INVALID_ARGUMENT", canonical(self.store.history(self.task["session_id"])))

    async def test_offline_demo_covers_resume_memory_and_confirmation(self):
        result = await demo(Path(self.temp.name) / "demo")
        self.assertEqual(result["kind"], "software_fixture")
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["writes_before_confirmation"], 0)
        self.assertEqual(result["writes_after_confirmation"], 1)
        self.assertEqual(result["recovered_wait_task"]["status"], "completed")
