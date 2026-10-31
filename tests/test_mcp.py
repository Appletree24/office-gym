import asyncio
import json
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.env.service import serve
from officegym.env.rollout import clone_world
from officegym.tools.dispatcher import Dispatcher


class MCPTests(unittest.TestCase):
    def test_stdio_list_write_and_replay(self):
        async def exercise(socket_path):
            from mcp import ClientSession,StdioServerParameters
            from mcp.client.stdio import stdio_client
            parameters = StdioServerParameters(command=sys.executable,args=['-m','officegym.tools.mcp_server','--socket',str(socket_path)],env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')})
            async with stdio_client(parameters) as (reader,writer):
                async with ClientSession(reader,writer) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    self.assertEqual(len(listing.tools),24)
                    self.assertTrue(all(tool.outputSchema for tool in listing.tools))
                    args = {'expected_version':1,'patch':{'goal':'schedule'}}
                    first = await session.call_tool('board.patch',args,meta={'officegym/operation_id':'mcp_write'})
                    self.assertFalse(first.isError,first)
                    self.assertEqual(first.content,[])
                    self.assertIsNotNone(first.structuredContent)
                    second = await session.call_tool('board.patch',args,meta={'officegym/operation_id':'mcp_write'})
                    self.assertEqual(first.structuredContent['data'],second.structuredContent['data'])
                    invalid = await session.call_tool('people.search',{'query':[]})
                    self.assertTrue(invalid.isError)
                    self.assertEqual(invalid.structuredContent['error']['code'],'INVALID_ARGUMENT')
                    return [first.structuredContent,second.structuredContent,invalid.structuredContent]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = World.create(root/'world.sqlite',sim_time=100,deadline=10000,principal_id='a')
            with world.db:
                world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',('a','a','a@x','UTC','[]'))
            direct = clone_world(root/'world.sqlite',root/'direct.sqlite')
            world.close()
            dispatcher = Dispatcher(direct)
            args = {'expected_version':1,'patch':{'goal':'schedule'}}
            expected = [dispatcher.call('board.patch',args,'mcp_write'),dispatcher.call('board.patch',args,'mcp_write'),dispatcher.call('people.search',{'query':[]})]
            process = multiprocessing.get_context('spawn').Process(target=serve,args=(root/'world.sqlite',root/'world.sock'))
            process.start()
            try:
                deadline = time.monotonic()+10
                while not (root/'world.sock').exists():
                    if time.monotonic()>deadline or not process.is_alive():
                        self.fail('world startup failed')
                    time.sleep(.01)
                actual = asyncio.run(asyncio.wait_for(exercise(root/'world.sock'),20))
                self.assertEqual(actual,expected)
                remote = World(root/'world.sqlite')
                try:
                    self.assertEqual(remote.state_hash(),direct.state_hash())
                    self.assertEqual(dispatcher.meta('tool_calls'),3)
                finally:
                    remote.close()
            finally:
                direct.close()
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
