"""Official SDK stdio adapter to the fixed-episode Unix world service."""
import argparse
import asyncio
from officegym.env.service import Client
from officegym.tools.output_schema import TOOL_RESULT_SCHEMA
from officegym.tools.schema import tool_schemas


def build_server(socket_path):
    from mcp.server.lowlevel import Server
    from mcp.types import Tool,CallToolResult
    server = Server('officegym')
    client = Client(socket_path)

    @server.list_tools()
    async def list_tools():
        return [Tool(**tool,outputSchema=TOOL_RESULT_SCHEMA) for tool in tool_schemas()]

    @server.call_tool(validate_input=False)
    async def call_tool(name,arguments):
        # Trusted operation keys travel in MCP request metadata, not model args.
        context = server.request_context
        metadata = context.meta.model_dump() if context.meta is not None else {}
        operation = metadata.get('officegym/operation_id')
        result = await asyncio.to_thread(client.call,name,arguments,operation)
        return CallToolResult(content=[],structuredContent=result,isError=not result['ok'])

    return server


async def run(socket_path):
    from mcp.server.stdio import stdio_server
    server = build_server(socket_path)
    async with stdio_server() as (reader,writer):
        await server.run(reader,writer,server.create_initialization_options())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket',required=True)
    args = parser.parse_args()
    asyncio.run(run(args.socket))
