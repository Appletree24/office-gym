import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World
from officegym.tools.dispatcher import Dispatcher


class AttachmentACLTests(unittest.TestCase):
    def test_private_reference_does_not_grant_recipient_access(self):
        with tempfile.TemporaryDirectory() as directory:
            world = World.create(Path(directory)/'w.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
            client.call('documents.put',dict(document_id='private',expected_version=0,title='Secret',content='secret'),'doc')
            args = dict(recipient_ids=['b'],kind='coordination',subject='Attachment',body='See attachment',attachment_refs=[{'document_id':'private','version':1}])
            denied = client.call('mail.send',args,'send')
            self.assertEqual(denied['error']['code'],'PERMISSION_DENIED')
            self.assertEqual(world.db.execute('SELECT count(*) FROM messages').fetchone()[0],0)
            self.assertEqual(world.db.execute('SELECT count(*) FROM outbox').fetchone()[0],0)
            self.assertFalse(client.call('operations.get',{'operation_id':'send'})['data']['committed'])
            # Trusted initial-world configuration grants access, not the mail tool.
            with world.db:
                world.db.execute('UPDATE documents SET acl_json=? WHERE document_id=?',('["a","b"]','private'))
            self.assertTrue(client.call('mail.send',args,'send')['ok'])
            self.assertEqual(world.db.execute('SELECT count(*) FROM outbox').fetchone()[0],1)
            world.close()
