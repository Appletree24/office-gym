import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class MailTests(unittest.TestCase):
    def test_atomic_outbox_replay_and_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for p in ('a','b','c'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(p,p,p+'@x','UTC','["*"]'))
            args = dict(recipient_ids=['b'],kind='coordination',subject='Review',body='Hello')
            first = client.call('mail.send',args,'one')
            self.assertTrue(first['ok'],first)
            self.assertEqual(client.call('mail.send',args,'one')['data'],first['data'])
            self.assertEqual(world.db.execute('SELECT count(*) FROM outbox').fetchone()[0],1)
            self.assertTrue(client.call('mail.send',args,'two')['ok'])
            self.assertEqual(world.db.execute('SELECT count(*) FROM outbox').fetchone()[0],2)
            self.assertEqual(world.db.execute('PRAGMA foreign_key_check').fetchall(),[])
            with world.db:
                client.set_meta('principal_id','c')
            self.assertEqual(client.call('mail.search',{'query':''})['data']['items'],[])
            world.close()
