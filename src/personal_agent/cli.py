"""Local assistant management, bounded scheduler and explicitly scripted demo."""
import argparse
import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import sys
import time
from .adapters import OfficeAdapter, connect_mcp
from .config import load
from .model import ConversationPolicy
from .models import AgentError, Turn, canonical
from .runtime import Runtime
from .store import Store, identity
from .tools import Registry


def initialize_office(path: Path):
    """Create fabricated office data, without an episode answer or scoring rules."""
    from officegym.env.world import World
    world = World.create(path, sim_time=int(time.time()), deadline=int(time.time()) + 86400,
                         principal_id="assistant")
    try:
        with world.db:
            world.db.execute("INSERT INTO people VALUES (?,?,?,?,?)",
                             ("assistant", "Demo assistant", "assistant@example.invalid", "UTC", '["assistant"]'))
            world.db.execute("INSERT INTO meta VALUES (?,?)", ("personal_world_id", canonical(identity("world"))))
    finally:
        world.close()


def report(store, task: dict):
    result = {key: task[key] for key in ("id", "session_id", "status", "instruction", "reason", "wait_until", "event_kind")}
    action = store.active_action(task["id"])
    if action:
        result["operation"] = {key: action[key] for key in ("id", "tool", "arguments", "status", "expires_at")}
    texts = [message["content"] for row in store.history(task["session_id"])
             if row["task_id"] == task["id"] for message in row["payload"]
             if message["role"] == "assistant" and not message.get("tool_call") and message.get("content")]
    if texts:
        result["answer"] = texts[-1]
    return result


class ScriptPolicy:
    """Finite software fixture used only by the offline demo."""
    policy_kind = "software_fixture"

    def __init__(self, turns):
        self.turns = iter(turns)

    def generate(self, messages, tools, remaining):
        turn = next(self.turns)
        return turn(messages) if callable(turn) else turn


async def demo(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "agent.sqlite").exists() or (root / "office.sqlite").exists():
        raise AgentError("DEMO_OUTPUT_EXISTS")
    initialize_office(root / "office.sqlite")
    now = [time.time()]
    store = Store(root / "agent.sqlite", "demo")
    office = OfficeAdapter(root / "office.sqlite", clock=lambda: now[0])
    try:
        registry = Registry()
        await registry.add("office", office)
        task = store.submit("记住我的称呼，下次对话也使用。", now[0])
        runtime = Runtime(store, ScriptPolicy([Turn(tool="task_ask", arguments={"question": "你希望我怎样称呼你？"})]), registry, clock=lambda: now[0])
        await runtime.run(task["id"])
        asking = store.task(task["id"])["status"]
        store.submit("请叫我小林。", now[0], task_id=task["id"])
        source = store.history(task["session_id"])[-1]["id"]
        runtime.policy = ScriptPolicy([
            Turn(tool="memory_put", arguments={"key": "preferred_name", "kind": "preference", "content": "小林", "source_ids": [source], "expected_version": 0}),
            Turn(content="已保存你的称呼，小林。")])
        await runtime.run(task["id"])
        wait = store.submit("收到 reply 事件后提醒我。", now[0], session_id=task["session_id"])
        runtime.policy = ScriptPolicy([Turn(tool="task_wait", arguments={"event_kind": "reply"})])
        await runtime.run(wait["id"])
        store.close()
        store = Store(root / "agent.sqlite", "demo")
        runtime = Runtime(store, ScriptPolicy([Turn(content="小林，等待的回复已经到达。")]), registry, clock=lambda: now[0])
        store.event("demo_reply", "reply", {"subject": "模拟回复"}, now[0])
        await runtime.tick()
        another = store.submit("在新会话里读取我的称呼。", now[0])
        runtime.policy = ScriptPolicy([Turn(tool="memory_search", arguments={"query": "preferred_name"}), Turn(content="欢迎回来，小林。")])
        await runtime.run(another["id"])
        writing = store.submit("保存一份演示文档。", now[0], session_id=another["session_id"])
        runtime.policy = ScriptPolicy([Turn(tool="office__documents_put", arguments={"document_id": "demo_note", "title": "演示笔记", "content": "仅为本地软件演示。", "expected_version": 0})])
        await runtime.run(writing["id"])
        action = store.active_action(writing["id"])
        before = office.world.db.execute("SELECT count(*) FROM documents").fetchone()[0]
        store.decide(action["id"], True, now[0])
        runtime.policy = ScriptPolicy([Turn(content="演示文档已写入模拟办公环境。")])
        await runtime.run(writing["id"])
        after = office.world.db.execute("SELECT count(*) FROM documents").fetchone()[0]
        result = {"kind": "software_fixture", "model_calls": 0, "clarification_status": asking,
                  "memory_across_sessions": store.memory_search("preferred_name"),
                  "recovered_wait_task": report(store, store.task(wait["id"])),
                  "writes_before_confirmation": before, "writes_after_confirmation": after,
                  "write_task": report(store, store.task(writing["id"]))}
        (root / "demo.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        store.close()
        office.close()


async def execute(args):
    if args.command == "demo":
        return await demo(Path(args.out))
    if args.command == "init-office":
        initialize_office(Path(args.path))
        return {"created": args.path, "kind": "simulated_office"}
    if not args.config:
        raise AgentError("CONFIG_REQUIRED")
    config = load(args.config)
    config.database.parent.mkdir(parents=True, exist_ok=True)
    store = Store(config.database, config.user_id)
    try:
        if args.command == "tasks":
            return [report(store, task) for task in store.tasks()]
        if args.command == "history":
            return store.history(args.session)
        if args.command == "memories":
            return store.memory_search(args.query)
        if args.command == "event":
            from .models import decode
            return store.event(args.id, args.kind, decode(args.payload), time.time())
        if args.command == "cancel":
            store.cancel(args.task, time.time())
            return report(store, store.task(args.task))
        if args.command == "operation":
            action = store.action(args.operation)
            return {key: action[key] for key in ("id", "task_id", "tool", "arguments", "status", "expires_at", "result")}
        no_run = getattr(args, "no_run", False)
        if not no_run and config.model is None:
            raise AgentError("MODEL_CONFIG_REQUIRED")
        if args.command in ("approve", "reject"):
            task_id = store.action(args.operation)["task_id"]
            store.decide(args.operation, args.command == "approve", time.time())
        elif args.command == "resume":
            task_id = args.task
            store.resume(task_id, time.time())
        elif args.command == "chat":
            task_id = store.submit(args.message, time.time(), args.session, args.task)["id"]
        if no_run:
            return report(store, store.task(task_id))
        async with AsyncExitStack() as stack:
            registry = Registry()
            for item in config.adapters:
                if item["kind"] == "officegym":
                    adapter = OfficeAdapter(item["path"])
                    stack.callback(adapter.close)
                else:
                    adapter = await stack.enter_async_context(connect_mcp(item))
                await registry.add(item["name"], adapter)
            runtime = Runtime(store, ConversationPolicy(**config.model), registry, config.budget)
            if args.command == "tick":
                return [report(store, task) for task in await runtime.tick(args.limit)]
            if args.command == "work":
                while True:
                    results = await runtime.tick(args.limit)
                    if results:
                        print(canonical([report(store, task) for task in results]), flush=True)
                    await asyncio.sleep(args.interval)
            return report(store, await runtime.run(task_id))
    finally:
        store.close()


def parser():
    p = argparse.ArgumentParser(description="Persistent personal assistant; OfficeGym evaluation remains available via officegym.")
    p.add_argument("--config", help="Local JSON configuration; paths are relative to this file.")
    commands = p.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="Start a task or answer a suspended task.")
    chat.add_argument("message")
    chat.add_argument("--session")
    chat.add_argument("--task")
    chat.add_argument("--no-run", action="store_true", help="Persist input without invoking a model.")
    for name in ("tasks",):
        commands.add_parser(name)
    history = commands.add_parser("history")
    history.add_argument("--session", required=True)
    memories = commands.add_parser("memories")
    memories.add_argument("--query", default="")
    for name in ("resume", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("task")
        if name == "resume":
            command.add_argument("--no-run", action="store_true")
    for name in ("approve", "reject", "operation"):
        command = commands.add_parser(name)
        command.add_argument("operation")
        if name != "operation":
            command.add_argument("--no-run", action="store_true")
    event = commands.add_parser("event")
    event.add_argument("--id", required=True)
    event.add_argument("--kind", required=True)
    event.add_argument("--payload", default="{}")
    for name in ("tick", "work"):
        command = commands.add_parser(name)
        command.add_argument("--limit", type=int, default=10)
        if name == "work":
            command.add_argument("--interval", type=float, default=5)
    example = commands.add_parser("demo")
    example.add_argument("--out", required=True)
    init = commands.add_parser("init-office")
    init.add_argument("--path", required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command == "work" and (not 0.1 <= args.interval <= 60):
        parser().error("interval must be between 0.1 and 60 seconds")
    try:
        print(canonical(asyncio.run(execute(args))))
    except KeyboardInterrupt:
        return
    except AgentError as exc:
        print(canonical({"ok": False, "error": {"code": exc.code}}), file=sys.stderr)
        raise SystemExit(1) from None
    except (ValueError, OSError):
        print(canonical({"ok": False, "error": {"code": "INVALID_CONFIG_OR_STORAGE"}}), file=sys.stderr)
        raise SystemExit(1) from None
    except Exception:
        print(canonical({"ok": False, "error": {"code": "COMMAND_FAILED"}}), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
