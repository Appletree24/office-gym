from pathlib import Path
import tempfile
import unittest
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class EventFilterTests(unittest.TestCase):
    def test_filtered_pages_advance_without_skipping_matching_or_exposing_hidden_events(self):
        with tempfile.TemporaryDirectory() as directory:
            world=World.create(Path(directory)/'world.sqlite',sim_time=100,deadline=10000,principal_id='a');client=Dispatcher(world)
            try:
                with world.db:
                    for index,kind in enumerate(('Notification','Other','Notification','Other','Notification'),1):
                        world.db.execute('INSERT INTO event_log(event_id,sim_time,actor_id,kind,details_json,visibility_json) VALUES (?,?,?,?,?,?)',(str(index),100,'environment',kind,'{}','["b"]' if index==5 else '["a"]'))
                first=client.call('events.poll',{'after_seq':0,'limit':1,'kinds':['Notification']});self.assertTrue(first['ok']);self.assertEqual([r['seq'] for r in first['data']['items']],[1]);self.assertEqual(first['data']['next_seq'],1)
                second=client.call('events.poll',{'after_seq':1,'limit':1,'kinds':['Notification']});self.assertEqual([r['seq'] for r in second['data']['items']],[3]);self.assertEqual(second['data']['next_seq'],4)
                empty=client.call('events.poll',{'after_seq':4,'kinds':['Notification']});self.assertEqual(empty['data'],{'items':[],'next_seq':4})
                for kinds in ([],['Notification','Notification'],['']):self.assertFalse(client.call('events.poll',{'after_seq':0,'kinds':kinds})['ok'])
                all_events=client.call('events.poll',{'after_seq':0});self.assertEqual([r['seq'] for r in all_events['data']['items']],[1,2,3,4])
            finally:world.close()
