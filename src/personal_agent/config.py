"""Strict local configuration; credentials remain environment references."""
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from urllib.parse import urlparse
from .models import Budget, decode


def endpoint(value):
    if not isinstance(value, str):
        raise ValueError("URL required")
    url = urlparse(value)
    if (url.username or url.password or not url.hostname or url.fragment
            or url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("127.0.0.1", "::1", "localhost"))):
        raise ValueError("HTTPS or local loopback HTTP URL required")
    return value


def keys(data, allowed):
    if not isinstance(data, dict) or set(data) - set(allowed):
        raise ValueError("unknown configuration fields")


def strings(data):
    if not isinstance(data, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in data.items()):
        raise ValueError("nonempty environment references required")


@dataclass(frozen=True)
class Config:
    user_id: str
    database: Path
    model: dict | None = None
    adapters: list[dict] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)


def load(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open("rb") as source:
        body = source.read(65537)
    if len(body) > 65536:
        raise ValueError("configuration too large")
    data = decode(body)
    keys(data, ("user_id", "database", "model", "adapters", "budget"))
    if not isinstance(data.get("user_id"), str) or not data["user_id"] or len(data["user_id"]) > 256:
        raise ValueError("user_id required")
    if not isinstance(data.get("database"), str) or not data["database"]:
        raise ValueError("database path required")
    model = data.get("model")
    if model is not None:
        keys(model, ("endpoint", "model", "model_version", "api_key_env", "max_output_tokens", "temperature"))
        endpoint(model.get("endpoint"))
        if any(not isinstance(model.get(key), str) or not model[key] for key in ("model", "model_version")):
            raise ValueError("model identity required")
        if "api_key_env" in model and (not isinstance(model["api_key_env"], str) or not model["api_key_env"]):
            raise ValueError("credential environment name required")
        cap = model.get("max_output_tokens", 1024)
        temperature = model.get("temperature", 0)
        if type(cap) is not int or cap < 1 or type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("invalid model limits")
    adapters = data.get("adapters", [])
    if not isinstance(adapters, list) or len(adapters) > 16:
        raise ValueError("bounded adapter list required")
    names = set()
    for adapter in adapters:
        keys(adapter, ("name", "kind", "path", "transport", "command", "args", "env_refs",
                       "url", "header_env_refs", "readonly_tools", "idempotent",
                       "operation_meta_key", "receipt_tool", "receipt_argument", "timeout_seconds"))
        name = adapter.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,15}", name) or name in names:
            raise ValueError("unique adapter name required")
        names.add(name)
        if adapter.get("kind") == "officegym":
            if not isinstance(adapter.get("path"), str) or not adapter["path"]:
                raise ValueError("office world path required")
            adapter["path"] = str((path.parent / adapter["path"]).resolve())
            if not Path(adapter["path"]).is_file():
                raise ValueError("office world does not exist")
        elif adapter.get("kind") == "mcp":
            if adapter.get("transport") == "stdio":
                if not isinstance(adapter.get("command"), str) or not adapter["command"]:
                    raise ValueError("MCP command required")
                args = adapter.get("args", [])
                if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
                    raise ValueError("MCP args must be strings")
                strings(adapter.get("env_refs", {}))
            elif adapter.get("transport") == "streamable-http":
                endpoint(adapter.get("url"))
                strings(adapter.get("header_env_refs", {}))
            else:
                raise ValueError("supported MCP transport required")
            readonly = adapter.get("readonly_tools", [])
            if not isinstance(readonly, list) or any(not isinstance(tool, str) or not tool for tool in readonly) or len(set(readonly)) != len(readonly):
                raise ValueError("readonly_tools must be unique names")
            if type(adapter.get("idempotent", False)) is not bool:
                raise ValueError("idempotent must be boolean")
            if adapter.get("idempotent") and not adapter.get("receipt_tool"):
                raise ValueError("idempotent retry requires a receipt tool")
            for key in ("receipt_tool", "receipt_argument", "operation_meta_key"):
                if key in adapter and (not isinstance(adapter[key], str) or not adapter[key]):
                    raise ValueError("receipt and metadata names must be nonempty")
            timeout = adapter.get("timeout_seconds", 30)
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("positive MCP timeout required")
        else:
            raise ValueError("supported adapter kind required")
    keys(data.get("budget", {}), Budget.__dataclass_fields__)
    return Config(data["user_id"], (path.parent / data["database"]).resolve(),
                  model, adapters, Budget(**data.get("budget", {})))
