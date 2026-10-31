import json
import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class ReplyTests(unittest.TestCase):
    def test_structured_request_and_delayed_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
                client.set_meta('reply_rules',{'b':{'availability':{'delay_seconds':120,'response':'provided','facts':{'available_windows':[[4000,6000]]}}}})
            args = dict(recipient_ids=['b'],kind='coordination',subject='When?',body='Please confirm immediately')
            self.assertTrue(client.call('mail.send',args,'plain')['ok'])
            self.assertEqual(world.db.execute('SELECT count(*) FROM scheduled_events').fetchone()[0],0)
            result = client.call('mail.send',dict(args,request={'type':'availability'}),'structured')
            self.assertTrue(result['ok'],result)
            self.assertEqual(world.db.execute("SELECT count(*) FROM messages WHERE kind='reply'").fetchone()[0],0)
            client.call('clock.wait',{'until':1000})
            thread = client.call('mail.get_thread',{'thread_id':result['data']['message_id']})['data']['items']
            self.assertEqual(len(thread),2)
            self.assertEqual(json.loads(thread[1]['body'])['facts']['available_windows'],[[4000,6000]])
            self.assertEqual(world.db.execute('SELECT count(*) FROM approvals').fetchone()[0],0)
            world.close()
