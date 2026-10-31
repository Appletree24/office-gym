import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from officegym.env.world import World,canonical
from officegym.env.events import PUBLIC_EVENT_FIELDS
from officegym.tools.dispatcher import Dispatcher
from officegym.tools.schema import TOOL_SCHEMA_VERSION


class PublicEventProjectionTests(unittest.TestCase):
    def test_notice_visibility_does_not_grant_document_content_access(self):
        with tempfile.TemporaryDirectory() as directory:
            world=World.create(Path(directory)/'world.sqlite',sim_time=100,deadline=10000,principal_id='a')
            client=Dispatcher(world)
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
                body='SECRET_OLD_DOCUMENT_BODY'
                world.db.execute('INSERT INTO documents VALUES (?,?,?,?,?,?,?,?)',('private',1,'Restricted',body,hashlib.sha256(body.encode()).hexdigest(),'b','["b"]',100))
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('hidden',150,1,'Notification','{}',canonical(dict(marker='HIDDEN_NOTIFICATION')),'["b"]','pending'))
                world.db.execute('INSERT INTO scheduled_events VALUES (?,?,?,?,?,?,?,?)',('publish',200,1,'DocumentVersionPublished','{}',canonical(dict(document_id='private',expected_version=1,content='SECRET_NEW_DOCUMENT_BODY')),'["a"]','pending'))
            client.call('clock.wait',dict(until=500))
            denied=client.call('documents.read',dict(document_id='private',version=2))
            self.assertEqual(denied['error']['code'],'PERMISSION_DENIED')
            response=client.call('events.poll',dict(after_seq=0))
            self.assertEqual(response['meta']['tool_schema_version'],TOOL_SCHEMA_VERSION)
            events=response['data']['items']
            self.assertEqual(len(events),1)
            self.assertEqual(set(events[0]),set(PUBLIC_EVENT_FIELDS))
            self.assertEqual(json.loads(events[0]['details_json']),dict(document_id='private',version=2))
            for secret in ('SECRET_OLD_DOCUMENT_BODY','SECRET_NEW_DOCUMENT_BODY','HIDDEN_NOTIFICATION'):
                self.assertNotIn(secret,canonical(response))
            audit=world.db.execute("SELECT * FROM event_log WHERE kind='DocumentVersionPublished'").fetchone()
            self.assertIn('SECRET_OLD_DOCUMENT_BODY',audit['before_json'])
            self.assertIn('SECRET_NEW_DOCUMENT_BODY',audit['after_json'])
            self.assertEqual(audit['seq'],events[0]['seq'])
            self.assertEqual(client.call('events.poll',dict(after_seq=response['data']['next_seq']))['data']['items'],[])
            world.close()
