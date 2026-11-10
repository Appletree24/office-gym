"""Business-only office simulation and configured real MCP application connections."""
from contextlib import asynccontextmanager, AsyncExitStack
from datetime import timedelta
import os
from pathlib import Path
from .models import AgentError, Receipt, ReceiptStatus, Tool, digest, utc_now


class OfficeAdapter:
    """Dedicated office world; synthetic data and effects never leave this file."""
    idempotent = True

    def __init__(self, path: str | Path, clock=utc_now):
        from officegym.env.world import World
        from officegym.tools.dispatcher import BusinessDispatcher
        self.world = World(path)
        self.dispatcher = BusinessDispatcher(self.world)
        if self.dispatcher.meta("principal_id") is None:
            self.world.close()
            raise AgentError("INVALID_OFFICE_WORLD")
        self.clock = clock
        self.fingerprint = digest(["office-business-v1", str(Path(path).resolve()),
                                   self.dispatcher.meta("principal_id"), self.dispatcher.meta("personal_world_id")])

    async def list_tools(self):
        from officegym.tools.schema import WRITES, tool_schemas
        from officegym.tools.dispatcher import APPLICATION_TOOLS
        return [Tool(tool["name"], "Simulated office: " + tool["description"],
                     tool["inputSchema"], "", tool["name"], tool["name"] not in WRITES)
                for tool in tool_schemas() if tool["name"] in APPLICATION_TOOLS]

    async def call(self, name, arguments, operation_id=None):
        return self.dispatcher.call(name, arguments, operation_id, now=int(self.clock()))

    async def reconcile(self, operation_id):
        result = self.dispatcher.call("operations.get", {"operation_id": operation_id}, now=int(self.clock()))
        if not result["ok"]:
            return Receipt(ReceiptStatus.UNKNOWN)
        data = result["data"]
        return Receipt(ReceiptStatus.COMMITTED, {"ok": True, "data": data["result"], "error": None}) if data["committed"] else Receipt(ReceiptStatus.NOT_COMMITTED)

    def close(self):
        self.world.close()


class MCPAdapter:
    """Read/write classification and receipt promises come from trusted configuration."""

    def __init__(self, session, config: dict, resolved_identity: dict):
        self.session, self.config = session, config
        self.idempotent = config.get("idempotent", False)
        self.fingerprint = digest(["mcp-v1", config, resolved_identity])

    async def list_tools(self):
        tools, cursor = [], None
        for _ in range(20):
            listing = await self.session.list_tools(cursor=cursor)
            tools.extend(listing.tools)
            cursor = listing.nextCursor
            if not cursor:
                break
        else:
            raise AgentError("MCP_TOOL_LIST_TOO_LARGE")
        if len(tools) > 256:
            raise AgentError("MCP_TOOL_LIST_TOO_LARGE")
        names = {tool.name for tool in tools}
        readonly = set(self.config.get("readonly_tools", []))
        lookup = self.config.get("receipt_tool")
        if readonly - names or lookup and lookup not in names:
            raise AgentError("MCP_CONFIG_TOOL_NOT_FOUND")
        return [Tool(tool.name, tool.description or tool.name, tool.inputSchema, "", tool.name, tool.name in readonly)
                for tool in tools if tool.name != lookup]

    def _result(self, response):
        if response.isError:
            return {"ok": False, "error": {"code": "APPLICATION_ERROR"}}
        if response.structuredContent is not None:
            return response.structuredContent
        # Unstructured tools have a single bounded representation, not duplicate content.
        blocks = []
        for content in response.content:
            if content.type != "text":
                return {"ok": False, "error": {"code": "UNSUPPORTED_MCP_CONTENT"}}
            blocks.append(content.text)
        return {"ok": True, "data": {"text": blocks}}

    async def call(self, name, arguments, operation_id=None):
        metadata = {self.config.get("operation_meta_key", "personal_agent/operation_id"): operation_id} if operation_id else None
        result = await self.session.call_tool(name, arguments, meta=metadata)
        return self._result(result)

    async def reconcile(self, operation_id):
        name = self.config.get("receipt_tool")
        if not name:
            return Receipt(ReceiptStatus.UNKNOWN)
        response = await self.session.call_tool(name, {self.config.get("receipt_argument", "operation_id"): operation_id})
        data = response.structuredContent
        if response.isError or not isinstance(data, dict) or type(data.get("committed")) is not bool:
            return Receipt(ReceiptStatus.UNKNOWN)
        if data["committed"]:
            if "result" not in data:
                return Receipt(ReceiptStatus.UNKNOWN)
            return Receipt(ReceiptStatus.COMMITTED, data["result"])
        return Receipt(ReceiptStatus.NOT_COMMITTED)


@asynccontextmanager
async def connect_mcp(config: dict):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    resolved = {}
    for key, env_name in config.get("env_refs", {}).items():
        if env_name not in os.environ:
            raise AgentError("APPLICATION_CREDENTIAL_UNSET")
        resolved[key] = os.environ[env_name]
    async with AsyncExitStack() as stack:
        if config["transport"] == "stdio":
            connection = stdio_client(StdioServerParameters(
                command=config["command"], args=config.get("args", []), env=resolved))
        else:
            import httpx
            headers = {}
            for key, env_name in config.get("header_env_refs", {}).items():
                if env_name not in os.environ:
                    raise AgentError("APPLICATION_CREDENTIAL_UNSET")
                headers[key] = os.environ[env_name]
            resolved = headers
            client = await stack.enter_async_context(httpx.AsyncClient(
                headers=headers, timeout=config.get("timeout_seconds", 30),
                follow_redirects=False, trust_env=False))
            connection = streamable_http_client(config["url"], http_client=client)
        streams = await stack.enter_async_context(connection)
        async with ClientSession(streams[0], streams[1],
                                 read_timeout_seconds=timedelta(seconds=config.get("timeout_seconds", 30))) as session:
            await session.initialize()
            yield MCPAdapter(session, config, resolved)
