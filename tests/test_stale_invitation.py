import tempfile
import unittest
from pathlib import Path
from officegym.env.world import World,canonical
from officegym.tools.dispatcher import Dispatcher
from officegym.graders.result import evaluate


class StaleInvitationTests(unittest.TestCase):
    def test_offline_stale_version_after_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'w.sqlite'
            world = World.create(path,sim_time=100,deadline=10000,principal_id='a')
            client = Dispatcher(world)
            with world.db:
                for person in ('a','b'):
                    world.db.execute('INSERT INTO people VALUES (?,?,?,?,?)',(person,person,person+'@x','UTC','["*"]'))
            first = client.call('calendar.create',dict(title='Review',participant_ids=['b'],start_at=3000,end_at=4800),'m')['data']
            client.call('calendar.update',dict(meeting_id=first['id'],expected_version=1,patch={'title':'Revised'}),'u')
            message = client.call('mail.send',dict(recipient_ids=['b'],kind='invitation',subject='Review',body='',meeting_id=first['id'],meeting_version=2),'s')['data']['message_id']
            # Offline corruption models a broken adapter bypassing dispatcher versions.
            with world.db:
                body = canonical({key:first[key] for key in ('id','version','title','start_at','end_at','participant_ids_json','document_refs_json','business_digest')})
                world.db.execute('UPDATE messages SET meeting_version=1,body=? WHERE id=?',(body,message))
                world.db.execute('UPDATE outbox SET meeting_version=1 WHERE message_id=?',(message,))
            spec = dict(principal_id='a',required_participant_ids=['b'],allowed_window_utc=[3000,8000],deadline_at=10000)
            self.assertIn('stale_meeting_version_sent',evaluate(path,spec)['hard_violations'])
            world.close()
