import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher
from officegym.training.tool_execution import dispatch_generated_call,single_call


class ToolExecutionTests(unittest.TestCase):
    def test_invalid_json_and_unknown_tool_are_billed_without_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'world.sqlite',sim_time=100,deadline=10000,principal_id='a')
            try:
                dispatcher = Dispatcher(world)
                bad_json = dispatch_generated_call(dispatcher,'calendar.create','{')
                unknown = dispatch_generated_call(dispatcher,'unknown.tool','{}')
                self.assertFalse(bad_json['ok'])
                self.assertFalse(unknown['ok'])
                self.assertEqual(dispatcher.meta('tool_calls'),2)
                self.assertEqual(dispatcher.meta('sim_time'),160)
                self.assertEqual(world.db.execute('SELECT count(*) FROM meetings').fetchone()[0],0)
                self.assertFalse(single_call([object(),object()]))
                self.assertTrue(single_call([object()]))
            finally:
                world.close()
