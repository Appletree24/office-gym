-- OfficeGym env-v1: one file per episode. Business logic is not implemented here.
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
CREATE TABLE meta (key TEXT PRIMARY KEY, value_json TEXT NOT NULL CHECK(json_valid(value_json)));
CREATE TABLE people (
 id TEXT PRIMARY KEY, display_name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
 timezone TEXT NOT NULL, visibility_json TEXT NOT NULL CHECK(json_valid(visibility_json))
);
CREATE TABLE documents (
 document_id TEXT NOT NULL, version INTEGER NOT NULL CHECK(version > 0),
 title TEXT NOT NULL, content TEXT NOT NULL, content_sha256 TEXT NOT NULL,
 owner_id TEXT NOT NULL REFERENCES people(id),
 acl_json TEXT NOT NULL CHECK(json_valid(acl_json)), created_at INTEGER NOT NULL,
 PRIMARY KEY(document_id, version)
);
CREATE TABLE meetings (
 id TEXT PRIMARY KEY, version INTEGER NOT NULL CHECK(version > 0),
 title TEXT NOT NULL, organizer_id TEXT NOT NULL REFERENCES people(id),
 participant_ids_json TEXT NOT NULL CHECK(json_valid(participant_ids_json)),
 start_at INTEGER NOT NULL, end_at INTEGER NOT NULL CHECK(end_at > start_at),
 status TEXT NOT NULL CHECK(status IN ('draft','scheduled','cancelled','completed')),
 document_refs_json TEXT NOT NULL CHECK(json_valid(document_refs_json)),
 business_digest TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE TABLE meeting_history (
 meeting_id TEXT NOT NULL REFERENCES meetings(id), version INTEGER NOT NULL,
 snapshot_json TEXT NOT NULL CHECK(json_valid(snapshot_json)),
 PRIMARY KEY(meeting_id, version)
);
CREATE TABLE approvals (
 id TEXT PRIMARY KEY, approver_id TEXT NOT NULL REFERENCES people(id),
 meeting_id TEXT NOT NULL REFERENCES meetings(id), meeting_version INTEGER NOT NULL,
 business_digest TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','granted','rejected','expired','invalidated')),
 requested_at INTEGER NOT NULL, resolved_at INTEGER, expires_at INTEGER,
 resolution_event_id TEXT
);
CREATE TABLE messages (
 id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, sender_id TEXT NOT NULL REFERENCES people(id),
 recipient_ids_json TEXT NOT NULL CHECK(json_valid(recipient_ids_json)),
 kind TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL,
 attachment_refs_json TEXT NOT NULL CHECK(json_valid(attachment_refs_json)),
 request_json TEXT CHECK(request_json IS NULL OR json_valid(request_json)),
 meeting_id TEXT REFERENCES meetings(id), meeting_version INTEGER, sent_at INTEGER NOT NULL
);
CREATE INDEX messages_thread_order ON messages(thread_id, sent_at, id);
CREATE TABLE todos (
 id TEXT PRIMARY KEY, version INTEGER NOT NULL CHECK(version > 0),
 title TEXT NOT NULL, assignee_id TEXT NOT NULL REFERENCES people(id),
 due_at INTEGER NOT NULL, source_meeting_id TEXT REFERENCES meetings(id),
 status TEXT NOT NULL CHECK(status IN ('open','blocked','done','cancelled')),
 created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
);
CREATE TABLE scheduled_events (
 id TEXT PRIMARY KEY, due_at INTEGER NOT NULL, priority INTEGER NOT NULL,
 kind TEXT NOT NULL, preconditions_json TEXT NOT NULL CHECK(json_valid(preconditions_json)),
 payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
 visibility_json TEXT NOT NULL CHECK(json_valid(visibility_json)),
 status TEXT NOT NULL CHECK(status IN ('pending','applied','cancelled'))
);
CREATE INDEX scheduled_events_due ON scheduled_events(status, due_at, priority, id);
CREATE TABLE operations (
 operation_id TEXT PRIMARY KEY, principal_id TEXT NOT NULL REFERENCES people(id),
 tool_name TEXT NOT NULL, request_sha256 TEXT NOT NULL,
 response_json TEXT NOT NULL CHECK(json_valid(response_json)), committed_at INTEGER NOT NULL
);
CREATE TABLE event_log (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, sim_time INTEGER NOT NULL,
 actor_id TEXT, cause_operation_id TEXT, cause_event_id TEXT,
 kind TEXT NOT NULL, object_type TEXT, object_id TEXT, business_version INTEGER,
 before_json TEXT CHECK(before_json IS NULL OR json_valid(before_json)),
 after_json TEXT CHECK(after_json IS NULL OR json_valid(after_json)),
 details_json TEXT NOT NULL CHECK(json_valid(details_json)),
 visibility_json TEXT NOT NULL CHECK(json_valid(visibility_json))
);
CREATE TRIGGER event_log_no_update BEFORE UPDATE ON event_log
 BEGIN SELECT RAISE(ABORT, 'event_log is append-only'); END;
CREATE TRIGGER event_log_no_delete BEFORE DELETE ON event_log
 BEGIN SELECT RAISE(ABORT, 'event_log is append-only'); END;
CREATE TABLE outbox (
 id TEXT PRIMARY KEY, operation_id TEXT NOT NULL REFERENCES operations(operation_id),
 message_id TEXT NOT NULL REFERENCES messages(id), recipient_id TEXT NOT NULL REFERENCES people(id),
 kind TEXT NOT NULL, meeting_id TEXT REFERENCES meetings(id), meeting_version INTEGER,
 canonical_purpose TEXT NOT NULL, sent_at INTEGER NOT NULL
);
-- Intentionally NOT UNIQUE: semantic duplicates must remain detectable by the grader.
CREATE INDEX outbox_semantic ON outbox(meeting_id, meeting_version, kind, recipient_id, canonical_purpose);
CREATE TABLE board (
 id INTEGER PRIMARY KEY CHECK(id = 1), version INTEGER NOT NULL,
 content_json TEXT NOT NULL CHECK(json_valid(content_json)), updated_at INTEGER NOT NULL
);

