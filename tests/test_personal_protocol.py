"""Real loopback HTTP and official MCP transports verify wire behavior."""
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from personal_agent.adapters import connect_mcp, OfficeAdapter
from personal_agent.cli import initialize_office
from personal_agent.config import load
from personal_agent.model import ConversationPolicy
from personal_agent.models import AgentError, Budget, ReceiptStatus, TaskStatus, Turn, canonical
from personal_agent.runtime import Runtime
from personal_agent.store import Store
from personal_agent.tools import Registry


class ModelHTTPTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.response = {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                owner.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = canonical(owner.response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.policy = ConversationPolicy(f"http://127.0.0.1:{self.server.server_port}/chat/completions",
                                         "fixture", "fixture-v1")
        self.remaining = {"generated_tokens": 10, "wall_seconds": 2}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def test_plain_text_and_function_calls_use_actual_http(self):
        turn = self.policy.generate([{"role": "user", "content": "hello"}], [], self.remaining)
        self.assertEqual(turn.content, "hello")
        self.assertEqual(turn.output_tokens, 3)
        self.assertFalse(self.requests[0]["parallel_tool_calls"])
        self.assertEqual(self.requests[0]["max_tokens"], 10)
        self.response["choices"][0]["message"] = {"content": None, "tool_calls": [
            {"id": "provider_call", "type": "function", "function": {"name": "task_wait", "arguments": '{"event_kind":"reply"}'}}]}
        self.response["choices"][0]["finish_reason"] = "tool_calls"
        turn = self.policy.generate([{"role": "user", "content": "wait"}], [], self.remaining)
        self.assertEqual(turn.arguments, {"event_kind": "reply"})
        self.assertEqual(turn.tool, "task_wait")

    def test_duplicate_json_arguments_multiple_calls_and_missing_usage_fail(self):
        call = {"type": "function", "function": {"name": "task_wait", "arguments": '{"until":1,"until":2}'}}
        self.response["choices"][0]["message"] = {"tool_calls": [call]}
        with self.assertRaisesRegex(AgentError, "INVALID_MODEL_RESPONSE"):
            self.policy.generate([], [], self.remaining)
        call["function"]["arguments"] = "{}"
        self.response["choices"][0]["message"]["tool_calls"] = [call, call]
        with self.assertRaises(AgentError):
            self.policy.generate([], [], self.remaining)
        self.response["choices"][0]["message"] = {"content": "text"}
        del self.response["usage"]
        with self.assertRaises(AgentError):
            self.policy.generate([], [], self.remaining)

    def test_bounded_response_and_real_runtime_conversation(self):
        small = ConversationPolicy(self.policy.endpoint, "fixture", "fixture-v1", max_response_bytes=20)
        with self.assertRaisesRegex(AgentError, "MODEL_RESPONSE_TOO_LARGE"):
            small.generate([], [], self.remaining)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite", "alice")
            try:
                task = store.submit("hello", 100)
                token = store.claim(task["id"], 100, 20)
                for i in range(20):
                    store.append(task["id"], token, [{"role": "assistant", "content": "history " + "x" * 1000}], 100)
                store.transition(task["id"], token, TaskStatus.PAUSED, 100)
                store.resume(task["id"], 100)
                result = asyncio.run(Runtime(store, self.policy, budget=Budget(context_bytes=10000)).run(task["id"]))
                self.assertEqual(result["status"], "completed")
                self.assertEqual(store.history(task["session_id"])[-1]["payload"][0]["content"], "hello")
                payload = self.requests[-1]
                self.assertLessEqual(len(canonical({"messages": payload["messages"], "tools": payload["tools"]}).encode()), 10000)
            finally:
                store.close()


class MCPProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, config):
        async with connect_mcp(config) as adapter:
            registry = Registry()
            await registry.add("fixture", adapter)
            self.assertTrue(registry.tools["fixture__echo"].readonly)
            self.assertFalse(registry.tools["fixture__write"].readonly)
            self.assertNotIn("fixture__receipt", registry.tools)
            self.assertEqual(await adapter.call("echo", {"value": "hello"}, None), {"value": "hello"})
            result = await adapter.call("write", {"value": "hello"}, "operation_fixture")
            self.assertEqual(result, {"ok": True, "data": {"value": "hello"}})
            self.assertEqual((await adapter.reconcile("operation_fixture")).status, ReceiptStatus.COMMITTED)
            self.assertEqual((await adapter.reconcile("missing")).status, ReceiptStatus.NOT_COMMITTED)
            self.assertEqual(await adapter.call("write", {"value": "hello"}, "operation_fixture"), result)

    async def test_official_sdk_stdio_metadata_and_receipts(self):
        config = {"transport": "stdio", "command": sys.executable,
                  "args": [str(Path(__file__).with_name("personal_mcp_fixture.py"))],
                  "readonly_tools": ["echo"], "idempotent": True, "receipt_tool": "receipt"}
        await asyncio.wait_for(self.exercise(config), 15)

    async def test_official_sdk_streamable_http(self):
        import uvicorn
        from tests.personal_mcp_fixture import application
        # Bind before starting the server; no free-port discovery race.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(application().streamable_http_app(),
                                               log_level="error", lifespan="on"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        self.fail("server did not start")
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(self.exercise({"transport": "streamable-http",
                                                  "url": f"http://127.0.0.1:{port}/mcp",
                                                  "readonly_tools": ["echo"],
                                                  "idempotent": True, "receipt_tool": "receipt"}), 15)
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, 5)
            sock.close()

    async def test_business_only_office_does_not_advance_episode_or_expose_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "office.sqlite"
            initialize_office(path)
            adapter = OfficeAdapter(path, lambda: 2000000000)
            try:
                now = adapter.dispatcher.meta("sim_time")
                adapter.world.db.execute("UPDATE meta SET value_json='true' WHERE key='finished'")
                adapter.world.db.commit()
                tools = await adapter.list_tools()
                names = {tool.name for tool in tools}
                self.assertNotIn("office.get_context", names)
                self.assertNotIn("task.finish", names)
                self.assertNotIn("clock.wait", names)
                args = {"document_id": "note", "title": "note", "content": "hello", "expected_version": 0}
                result = await adapter.call("documents.put", args, "op")
                self.assertTrue(result["ok"])
                self.assertEqual(await adapter.call("documents.put", args, "op"), result)
                self.assertEqual(adapter.dispatcher.meta("sim_time"), now)
                self.assertTrue(adapter.dispatcher.meta("finished"))
                self.assertEqual((await adapter.reconcile("op")).status, ReceiptStatus.COMMITTED)
                self.assertEqual(adapter.world.db.execute("SELECT count(*) FROM documents").fetchone()[0], 1)
            finally:
                adapter.close()


class ConfigTests(unittest.TestCase):
    def test_missing_referents_unknown_fields_and_insecure_urls_fail_at_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            base = {"user_id": "alice", "database": "agent.sqlite"}
            path.write_text(canonical(base))
            self.assertEqual(load(path).database, Path(directory) / "agent.sqlite")
            for extra in ({"unknown": 1}, {"budget": {"lease_seconds": 10}},
                          {"adapters": [{"name": "office", "kind": "officegym", "path": "missing.sqlite"}]},
                          {"model": {"endpoint": "http://remote.example/chat", "model": "m", "model_version": "v"}},
                          {"adapters": [{"name": "app", "kind": "mcp", "transport": "stdio", "command": "test", "idempotent": True}]}):
                path.write_text(canonical(dict(base, **extra)))
                with self.assertRaises(ValueError):
                    load(path)
