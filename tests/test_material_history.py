import json
import sqlite3
import unittest
from officegym.graders.material_history import required_refs_at_send


class MaterialHistoryTests(unittest.TestCase):
    def test_future_publication_does_not_retroactively_change_required_version(self):
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
        db.executescript('CREATE TABLE documents(document_id TEXT,version INTEGER,created_at INTEGER);CREATE TABLE event_log(seq INTEGER,object_type TEXT,object_id TEXT,after_json TEXT,details_json TEXT);')
        db.executemany('INSERT INTO documents VALUES (?,?,?)',[('agenda',1,100),('agenda',2,100)])
        db.execute('INSERT INTO event_log VALUES (?,?,?,?,?)',(20,'document','agenda',json.dumps({'version':2}),'{}'))
        spec={'track_document_ids':['agenda'],'required_document_refs':[{'document_id':'static','version':3}]}
        self.assertEqual(required_refs_at_send(db,spec,10,500),[{'document_id':'static','version':3},{'document_id':'agenda','version':1}])
        self.assertEqual(required_refs_at_send(db,spec,21,500),[{'document_id':'static','version':3},{'document_id':'agenda','version':2}])
        db.close()
