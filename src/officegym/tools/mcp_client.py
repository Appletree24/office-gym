"""Harness-side MCP client wrapper; consume structured observations once."""
from contextlib import asynccontextmanager
import sys


class MCPEnvironment:
    def __init__(self,session):
        self.session = session

    async def call(self,name,arguments,operation_id=None):
        metadata = {'officegym/operation_id':operation_id} if operation_id is not None else None
        response = await self.session.call_tool(name,arguments,meta=metadata)
        if response.structuredContent is None:
            raise RuntimeError('world MCP response omitted structured observation')
        # No fallback concatenation of text and structuredContent.
        return response.structuredContent

    async def list_tools(self):
        response = await self.session.list_tools()
        return [tool.model_dump(exclude_none=True) for tool in response.tools]


@asynccontextmanager
async def connect(socket_path):
    from mcp import ClientSession,StdioServerParameters
    from mcp.client.stdio import stdio_client
    parameters = StdioServerParameters(command=sys.executable,args=['-m','officegym.tools.mcp_server','--socket',str(socket_path)])
    async with stdio_client(parameters) as (reader,writer):
        async with ClientSession(reader,writer) as session:
            await session.initialize()
            yield MCPEnvironment(session)
