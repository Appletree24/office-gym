"""Controlled official-SDK application; never contacts a real user or vendor."""
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import CallToolResult


def application():
    app = FastMCP("personal-agent-test", log_level="ERROR", json_response=True, stateless_http=True)
    effects = {}

    @app.tool()
    def echo(value: str) -> CallToolResult:
        """Return a supplied fixture value without side effects."""
        return CallToolResult(content=[], structuredContent={"value": value})

    @app.tool()
    def write(value: str, ctx: Context) -> CallToolResult:
        """Record a fixture effect by runtime operation metadata."""
        meta = ctx.request_context.meta
        operation = meta.model_dump().get("personal_agent/operation_id") if meta else None
        if not operation:
            return CallToolResult(content=[], structuredContent={"ok": False}, isError=True)
        result = effects.setdefault(operation, {"ok": True, "data": {"value": value}})
        return CallToolResult(content=[], structuredContent=result)

    @app.tool()
    def receipt(operation_id: str) -> CallToolResult:
        """Return the authoritative in-process fixture receipt."""
        return CallToolResult(content=[], structuredContent={
            "committed": operation_id in effects, "result": effects.get(operation_id)})

    return app


if __name__ == "__main__":
    application().run(transport="stdio")
