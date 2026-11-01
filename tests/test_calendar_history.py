import json
import sqlite3
import unittest
from officegym.graders.calendar_history import conflicts_at_send


class CalendarHistoryTests(unittest.TestCase):
    def test_send_boundary_uses_past_state_not_future_creation_or_cancellation(self):
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
        db.executescript('CREATE TABLE meeting_history(meeting_id TEXT,version INTEGER,snapshot_json TEXT);CREATE TABLE event_log(seq INTEGER,object_type TEXT,object_id TEXT,before_json TEXT,after_json TEXT);')
        busy=dict(id='busy',participant_ids_json='["b"]',organizer_id='c',status='draft',created_at=100,start_at=1000,end_at=2000)
        snapshot=dict(id='target',participant_ids_json='["b"]',start_at=1500,end_at=2500)
        db.execute('INSERT INTO meeting_history VALUES (?,?,?)',('busy',1,json.dumps(busy)))
        db.execute('INSERT INTO event_log VALUES (?,?,?,?,?)',(10,'meeting','busy',None,json.dumps(busy)))
        self.assertEqual(conflicts_at_send(db,snapshot,5,500),[])
        self.assertEqual(conflicts_at_send(db,snapshot,11,500),['busy'])
        cancelled={**busy,'status':'cancelled'}
        db.execute('INSERT INTO event_log VALUES (?,?,?,?,?)',(20,'meeting','busy',json.dumps(busy),json.dumps(cancelled)))
        self.assertEqual(conflicts_at_send(db,snapshot,15,500),['busy'])
        self.assertEqual(conflicts_at_send(db,snapshot,21,500),[])
        db.close()

    def test_later_reschedule_cannot_erase_or_retroactively_create_conflict(self):
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
        db.executescript('CREATE TABLE meeting_history(meeting_id TEXT,version INTEGER,snapshot_json TEXT);CREATE TABLE event_log(seq INTEGER,object_type TEXT,object_id TEXT,before_json TEXT,after_json TEXT);')
        busy=dict(id='busy',participant_ids_json='["b"]',organizer_id='c',status='draft',created_at=100,start_at=1000,end_at=2000)
        free={**busy,'start_at':3000,'end_at':4000}
        target=dict(id='target',participant_ids_json='["b"]',start_at=1500,end_at=2500)
        db.execute('INSERT INTO meeting_history VALUES (?,?,?)',('busy',1,json.dumps(busy)))
        db.execute('INSERT INTO event_log VALUES (?,?,?,?,?)',(20,'meeting','busy',json.dumps(busy),json.dumps(free)))
        self.assertEqual(conflicts_at_send(db,target,10,500),['busy'])
        self.assertEqual(conflicts_at_send(db,target,21,500),[])
        db.execute('INSERT INTO event_log VALUES (?,?,?,?,?)',(30,'meeting','busy',json.dumps(free),json.dumps(busy)))
        self.assertEqual(conflicts_at_send(db,target,25,500),[])
        self.assertEqual(conflicts_at_send(db,target,31,500),['busy'])
        db.close()

    def test_seeded_history_respects_creation_time_participants_and_touching_intervals(self):
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
        db.executescript('CREATE TABLE meeting_history(meeting_id TEXT,version INTEGER,snapshot_json TEXT);CREATE TABLE event_log(seq INTEGER,object_type TEXT,object_id TEXT,before_json TEXT,after_json TEXT);')
        busy=dict(id='busy',participant_ids_json='["c"]',organizer_id='b',status='draft',created_at=100,start_at=1000,end_at=2000)
        target=dict(id='target',participant_ids_json='["b"]',start_at=1500,end_at=2500)
        db.execute('INSERT INTO meeting_history VALUES (?,?,?)',('busy',1,json.dumps(busy)))
        self.assertEqual(conflicts_at_send(db,target,10,99),[])
        self.assertEqual(conflicts_at_send(db,target,10,100),['busy'])
        self.assertEqual(conflicts_at_send(db,{**target,'start_at':2000},10,100),[])
        self.assertEqual(conflicts_at_send(db,{**target,'participant_ids_json':'["other"]'},10,100),[])
        db.close()
