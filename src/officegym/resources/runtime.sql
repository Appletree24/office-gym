-- Agent-local durable state. Separate from the environment's transaction.
PRAGMA journal_mode = WAL;
CREATE TABLE runtime_meta (key TEXT PRIMARY KEY, value_json TEXT NOT NULL CHECK(json_valid(value_json)));
CREATE TABLE write_intents (
 operation_id TEXT PRIMARY KEY, action_seq INTEGER NOT NULL UNIQUE,
 tool_name TEXT NOT NULL, args_json TEXT NOT NULL CHECK(json_valid(args_json)),
 request_sha256 TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('PREPARED','ACKED','RECONCILED')),
 result_json TEXT CHECK(result_json IS NULL OR json_valid(result_json)),
 created_at_wall TEXT NOT NULL, acknowledged_at_wall TEXT
);
CREATE TABLE checkpoints (
 id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT NOT NULL,
 event_cursor INTEGER NOT NULL, sim_time INTEGER NOT NULL,
 state TEXT NOT NULL, payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
 content_sha256 TEXT NOT NULL
);
CREATE TABLE context_segments (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
 role TEXT NOT NULL, content_json TEXT NOT NULL CHECK(json_valid(content_json)),
 token_count INTEGER NOT NULL, source_refs_json TEXT NOT NULL CHECK(json_valid(source_refs_json))
);

