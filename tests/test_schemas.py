import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.tools.schema import tool_schemas


class SchemaTests(unittest.TestCase):
    def test_all_tools_and_invalid_arguments(self):
        self.assertEqual(len(tool_schemas()),24)
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            for name,args in [('people.search',{'query':[]}),('board.read',{'principal_id':'b'}),('calendar.get',{'meeting_id':{}}),('events.poll',{'after_seq':True}),('mail.send',{}),('does.not.exist',{})]:
                result = client.call(name,args)
                self.assertEqual(result['error']['code'],'INVALID_ARGUMENT',result)
            self.assertEqual(client.meta('tool_calls'),6)
            self.assertEqual(client.meta('sim_time'),280)
            world.close()
