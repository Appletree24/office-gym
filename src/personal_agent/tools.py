"""Actual registered tool descriptions and local task/memory operations."""
import re
from typing import Any
from .models import Adapter, AgentError, TaskStatus, Tool, canonical, digest


def obj(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


def text(maximum: int = 4096) -> dict:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


VERSION = {"type": "integer", "minimum": 0}
TIME = {"type": "number", "minimum": 0}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 100}
BUILTINS = [
    Tool("memory_search", "Find this user's saved facts and preferences across conversations.",
         obj({"query": {"type": "string", "maxLength": 256}, "limit": LIMIT}, ["query"]), "", ""),
    Tool("memory_put", "Save a fact or preference from real message source IDs. Creation needs expected_version=0; replacement needs the current version. Preferences require a user message source.",
         obj({"key": text(128), "kind": {"enum": ["fact", "preference"]}, "content": text(),
              "source_ids": {"type": "array", "items": text(128), "minItems": 1, "maxItems": 8, "uniqueItems": True},
              "expected_version": VERSION}), "", "", False),
    Tool("memory_delete", "Delete a saved memory using its current version.",
         obj({"key": text(128), "expected_version": {"type": "integer", "minimum": 1}}), "", "", False),
    Tool("task_ask", "Ask a clarification question and suspend this task until the user replies.",
         obj({"question": text()}), "", "", False),
    Tool("task_wait", "Suspend this task until a future UTC Unix timestamp or a NEW event of event_kind. Supplying both wakes on the first condition. Does not poll or advance simulated time.",
         obj({"until": TIME, "event_kind": text(128)}, []), "", "", False),
    Tool("task_finish", "Complete only the current task and present the answer to the user.",
         obj({"answer": text(16384)}), "", "", False),
    Tool("task_schedule", "Create a separate task to run at a future UTC Unix timestamp.",
         obj({"instruction": text(16384), "until": TIME}), "", "", False),
    Tool("task_cancel", "Cancel another task owned by this user; cannot cancel the current task.",
         obj({"task_id": text(128)}), "", "", False),
    Tool("task_list", "List this user's recent tasks and their current status.",
         obj({"limit": LIMIT}, []), "", ""),
    Tool("operation_get", "Read a journaled application operation's status and exact confirmation request. This tool cannot approve it.",
         obj({"operation_id": text(128)}), "", ""),
]


class Registry:
    def __init__(self):
        self.tools = {tool.name: tool for tool in BUILTINS}
        self.adapters: dict[str, Adapter] = {}
        self.bindings: dict[str, str] = {}

    async def add(self, name: str, adapter: Adapter):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,15}", name) or name in self.adapters:
            raise ValueError("unique adapter name required")
        from jsonschema import Draft202012Validator
        for remote in await adapter.list_tools():
            alias = name + "__" + re.sub(r"[^A-Za-z0-9_-]", "_", remote.remote_name)
            if len(alias) > 64 or alias in self.tools:
                raise ValueError("tool alias collision or excessive length")
            Draft202012Validator.check_schema(remote.schema)
            tool = Tool(alias, remote.description, remote.schema, name,
                        remote.remote_name, remote.readonly)
            self.tools[alias] = tool
            self.bindings[alias] = digest([adapter.fingerprint, tool.public(),
                                          tool.remote_name, tool.readonly])
        self.adapters[name] = adapter

    def validate(self, name: str, arguments: Any) -> Tool:
        from jsonschema import Draft202012Validator
        tool = self.tools.get(name)
        if not tool:
            raise AgentError("UNKNOWN_TOOL")
        if not isinstance(arguments, dict):
            raise AgentError("INVALID_ARGUMENT")
        try:
            canonical(arguments)
            Draft202012Validator(tool.schema).validate(arguments)
        except Exception as exc:
            raise AgentError("INVALID_ARGUMENT") from exc
        return tool

    def public(self):
        return [tool.public() for tool in self.tools.values()]


def builtin(store, tid: str, name: str, args: dict, now: float):
    """Called inside the runtime's lease-checked transaction; returns data and suspension."""
    status, wait = None, {}
    if name == "memory_search":
        data = store.memory_search(args["query"], args.get("limit", 20))
    elif name == "memory_put":
        data = store.put_memory(now=now, **args)
    elif name == "memory_delete":
        data = store.delete_memory(args["key"], args["expected_version"])
    elif name == "task_list":
        data = [{key: task[key] for key in ("id", "session_id", "instruction", "status", "wait_until", "reason")}
                for task in store.tasks(args.get("limit", 20))]
    elif name == "operation_get":
        action = store.action(args["operation_id"])
        data = {key: action[key] for key in ("id", "task_id", "tool", "arguments", "status", "expires_at", "result")}
    elif name == "task_ask":
        data, status = {"question": args["question"]}, TaskStatus.WAITING_USER
    elif name == "task_finish":
        data, status = {"answer": args["answer"]}, TaskStatus.COMPLETED
    elif name in ("task_wait", "task_schedule"):
        until, kind = args.get("until"), args.get("event_kind")
        if (until is None and kind is None) or (until is not None and until <= now):
            raise AgentError("INVALID_WAIT")
        if name == "task_schedule":
            data = store.schedule(tid, args["instruction"], until, now)
        else:
            data, status = {"waiting": True, "until": until, "event_kind": kind}, TaskStatus.WAITING_EVENT
            wait = {"wait_until": until, "event_kind": kind}
    elif name == "task_cancel":
        if args["task_id"] == tid:
            raise AgentError("INVALID_ARGUMENT")
        store._cancel(args["task_id"], now)
        data = {"cancelled": args["task_id"]}
    else:
        raise AgentError("UNKNOWN_TOOL")
    return data, status, wait
