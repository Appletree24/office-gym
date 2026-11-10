"""User-scoped SQLite state and atomic lease, confirmation and inbox transitions."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import uuid
from typing import Any
from .models import AgentError, ActionStatus, TaskStatus, TERMINAL, canonical, digest


SCHEMA_VERSION = 1
DDL = """
CREATE TABLE sessions (
 id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE tasks (
 id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL REFERENCES sessions(id),
 instruction TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL,
 updated_at REAL NOT NULL, lease_token TEXT, lease_until REAL,
 wait_until REAL, event_kind TEXT, after_event INTEGER, reason TEXT
);
CREATE INDEX tasks_schedule ON tasks(user_id,status,wait_until);
CREATE TABLE messages (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
 user_id TEXT NOT NULL, session_id TEXT NOT NULL REFERENCES sessions(id),
 task_id TEXT NOT NULL REFERENCES tasks(id), payload TEXT NOT NULL,
 user_source INTEGER NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE memories (
 user_id TEXT NOT NULL, key TEXT NOT NULL, kind TEXT NOT NULL,
 content TEXT NOT NULL, version INTEGER NOT NULL, source_ids TEXT NOT NULL,
 updated_at REAL NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(user_id,key)
);
CREATE TABLE actions (
 id TEXT PRIMARY KEY, user_id TEXT NOT NULL, task_id TEXT NOT NULL REFERENCES tasks(id),
 tool TEXT NOT NULL, arguments TEXT NOT NULL, fingerprint TEXT NOT NULL,
 call_json TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL,
 expires_at REAL NOT NULL, request_sha256 TEXT NOT NULL, result TEXT
);
CREATE INDEX actions_task ON actions(user_id,task_id,status);
CREATE TABLE events (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL, user_id TEXT NOT NULL,
 kind TEXT NOT NULL, payload TEXT NOT NULL, created_at REAL NOT NULL,
 UNIQUE(user_id,id)
);
"""


def identity(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex


class Store:
    """One connection per runtime; all mutations execute on its owning thread."""

    def __init__(self, path: str | Path, user_id: str):
        if not isinstance(user_id, str) or not user_id or len(user_id) > 256:
            raise ValueError("user_id must be nonempty and bounded")
        self.user_id = user_id
        self.db = sqlite3.connect(path, timeout=5, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA busy_timeout=5000")
            with self.transaction():
                version = self.db.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    for statement in DDL.split(";"):
                        if statement.strip():
                            self.db.execute(statement)
                    self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                elif version != SCHEMA_VERSION:
                    raise AgentError("UNSUPPORTED_DATABASE_VERSION")
        except BaseException:
            self.db.close()
            raise

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def close(self):
        self.db.close()

    def session(self, session_id: str):
        row = self.db.execute("SELECT * FROM sessions WHERE id=? AND user_id=?",
                              (session_id, self.user_id)).fetchone()
        if row is None:
            raise AgentError("NOT_FOUND")
        return dict(row)

    def new_session(self, now: float) -> str:
        sid = identity("session")
        self.db.execute("INSERT INTO sessions VALUES (?,?,?)", (sid, self.user_id, now))
        return sid

    def _message(self, task, payload: list[dict[str, Any]], now: float,
                 user_source: bool = False) -> str:
        mid = identity("message")
        self.db.execute("INSERT INTO messages(id,user_id,session_id,task_id,payload,user_source,created_at) VALUES (?,?,?,?,?,?,?)",
                        (mid, self.user_id, task["session_id"], task["id"],
                         canonical(payload), int(user_source), now))
        return mid

    def submit(self, text: str, now: float, session_id: str | None = None,
               task_id: str | None = None) -> dict:
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 16384:
            raise AgentError("INVALID_MESSAGE")
        with self.transaction():
            if task_id:
                task = self.task(task_id)
                if session_id and session_id != task["session_id"]:
                    raise AgentError("NOT_FOUND")
                if task["status"] not in (TaskStatus.WAITING_USER, TaskStatus.PAUSED, TaskStatus.READY):
                    raise AgentError("TASK_STATE_CONFLICT")
                action = self.active_action(task_id)
                if action:
                    if action["status"] != ActionStatus.APPROVED:
                        raise AgentError("ACTION_STATE_CONFLICT")
                    result = {"ok": False, "error": {"code": "USER_INPUT_CHANGED"}}
                    self.db.execute("UPDATE actions SET status=?,result=? WHERE id=? AND user_id=?",
                                    (ActionStatus.REJECTED, canonical(result), action["id"], self.user_id))
                    self._action_result(action, result, now)
                self._state(task_id, TaskStatus.READY, now)
            else:
                sid = session_id or self.new_session(now)
                self.session(sid)
                tid = identity("task")
                self.db.execute("INSERT INTO tasks(id,user_id,session_id,instruction,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                                (tid, self.user_id, sid, text, TaskStatus.READY, now, now))
                task = self.task(tid)
            self._message(task, [{"role": "user", "content": text}], now, True)
            return self.task(task["id"])

    def task(self, task_id: str) -> dict:
        row = self.db.execute("SELECT * FROM tasks WHERE id=? AND user_id=?",
                              (task_id, self.user_id)).fetchone()
        if row is None:
            raise AgentError("NOT_FOUND")
        return dict(row)

    def tasks(self, limit: int = 100) -> list[dict]:
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM tasks WHERE user_id=? ORDER BY created_at DESC,id LIMIT ?",
            (self.user_id, limit))]

    def _state(self, tid: str, status: TaskStatus, now: float, reason: str | None = None,
               *, wait_until: float | None = None, event_kind: str | None = None):
        seq = self.db.execute("SELECT coalesce(max(seq),0) FROM events WHERE user_id=?",
                              (self.user_id,)).fetchone()[0]
        self.db.execute("UPDATE tasks SET status=?,updated_at=?,reason=?,lease_token=NULL,lease_until=NULL,wait_until=?,event_kind=?,after_event=? WHERE id=? AND user_id=?",
                        (status, now, reason, wait_until, event_kind, seq, tid, self.user_id))

    def guard(self, tid: str, token: str, now: float):
        task = self.task(tid)
        if (task["status"] != TaskStatus.RUNNING or task["lease_token"] != token
                or task["lease_until"] <= now):
            raise AgentError("STALE_LEASE")
        return task

    def claim(self, tid: str, now: float, ttl: float) -> str | None:
        with self.transaction():
            task = self.task(tid)
            reclaim = task["status"] == TaskStatus.RUNNING and task["lease_until"] <= now
            if task["status"] != TaskStatus.READY and not reclaim:
                return None
            token = identity("lease")
            self.db.execute("UPDATE tasks SET status=?,lease_token=?,lease_until=?,updated_at=? WHERE id=? AND user_id=?",
                            (TaskStatus.RUNNING, token, now + ttl, now, tid, self.user_id))
            return token

    def transition(self, tid: str, token: str, status: TaskStatus, now: float,
                   reason: str | None = None, payload: list[dict] | None = None,
                   **wait):
        with self.transaction():
            task = self.guard(tid, token, now)
            if payload:
                self._message(task, payload, now)
            self._state(tid, status, now, reason, **wait)

    def append(self, tid: str, token: str, payload: list[dict], now: float):
        with self.transaction():
            self._message(self.guard(tid, token, now), payload, now)

    def resume(self, tid: str, now: float):
        with self.transaction():
            task = self.task(tid)
            if task["status"] not in (TaskStatus.PAUSED, TaskStatus.BLOCKED):
                raise AgentError("TASK_STATE_CONFLICT")
            self._state(tid, TaskStatus.READY, now)

    def cancel(self, tid: str, now: float):
        with self.transaction():
            self._cancel(tid, now)

    def _cancel(self, tid: str, now: float):
        task = self.task(tid)
        if task["status"] in TERMINAL:
            raise AgentError("TASK_STATE_CONFLICT")
        self._state(tid, TaskStatus.CANCELLED, now)
        self.db.execute("UPDATE actions SET status=? WHERE task_id=? AND user_id=? AND status IN (?,?)",
                        (ActionStatus.REJECTED, tid, self.user_id, ActionStatus.PENDING, ActionStatus.APPROVED))
        self.db.execute("UPDATE actions SET status=? WHERE task_id=? AND user_id=? AND status=?",
                        (ActionStatus.UNKNOWN, tid, self.user_id, ActionStatus.EXECUTING))

    def history(self, session_id: str, limit: int = 200) -> list[dict]:
        self.session(session_id)
        rows = self.db.execute("SELECT * FROM (SELECT * FROM messages WHERE session_id=? AND user_id=? ORDER BY seq DESC LIMIT ?) ORDER BY seq",
                               (session_id, self.user_id, limit)).fetchall()
        return [dict(row, payload=json.loads(row["payload"])) for row in rows]

    def memory_search(self, query: str = "", limit: int = 20) -> list[dict]:
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self.db.execute("SELECT * FROM memories WHERE user_id=? AND deleted=0 AND (key LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\') ORDER BY updated_at DESC,key LIMIT ?",
                               (self.user_id, "%" + escaped + "%", "%" + escaped + "%", limit))
        return [dict(row, source_ids=json.loads(row["source_ids"])) for row in rows]

    def put_memory(self, key: str, kind: str, content: str, source_ids: list[str],
                   expected_version: int, now: float) -> dict:
        if kind not in ("preference", "fact") or not source_ids:
            raise AgentError("INVALID_ARGUMENT")
        user_source = False
        for mid in source_ids:
            row = self.db.execute("SELECT user_source FROM messages WHERE id=? AND user_id=?",
                                  (mid, self.user_id)).fetchone()
            if row is None:
                raise AgentError("INVALID_SOURCE")
            user_source |= bool(row[0])
        if kind == "preference" and not user_source:
            raise AgentError("USER_SOURCE_REQUIRED")
        row = self.db.execute("SELECT version,deleted FROM memories WHERE user_id=? AND key=?",
                              (self.user_id, key)).fetchone()
        if expected_version != (row["version"] if row and not row["deleted"] else 0):
            raise AgentError("VERSION_CONFLICT")
        version = row["version"] + 1 if row else 1
        self.db.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?,0) ON CONFLICT(user_id,key) DO UPDATE SET kind=excluded.kind,content=excluded.content,version=excluded.version,source_ids=excluded.source_ids,updated_at=excluded.updated_at,deleted=0",
                        (self.user_id, key, kind, content, version, canonical(source_ids), now))
        return {"key": key, "version": version}

    def delete_memory(self, key: str, version: int):
        row = self.db.execute("SELECT version,deleted FROM memories WHERE user_id=? AND key=?",
                              (self.user_id, key)).fetchone()
        if not row or row["deleted"]:
            raise AgentError("NOT_FOUND")
        if row[0] != version:
            raise AgentError("VERSION_CONFLICT")
        self.db.execute("UPDATE memories SET content='',source_ids='[]',deleted=1,version=version+1 WHERE user_id=? AND key=?",
                        (self.user_id, key))
        return {"deleted": True}

    def schedule(self, tid: str, instruction: str, until: float, now: float) -> dict:
        task = self.task(tid)
        child = identity("task")
        self.db.execute("INSERT INTO tasks(id,user_id,session_id,instruction,status,created_at,updated_at,wait_until) VALUES (?,?,?,?,?,?,?,?)",
                        (child, self.user_id, task["session_id"], instruction, TaskStatus.WAITING_EVENT, now, now, until))
        self._message(self.task(child), [{"role": "user", "content": instruction}], now)
        return {"task_id": child, "wake_at": until}

    def action(self, aid: str) -> dict:
        row = self.db.execute("SELECT * FROM actions WHERE id=? AND user_id=?",
                              (aid, self.user_id)).fetchone()
        if not row:
            raise AgentError("NOT_FOUND")
        return dict(row, arguments=json.loads(row["arguments"]),
                    call_json=json.loads(row["call_json"]),
                    result=json.loads(row["result"]) if row["result"] else None)

    def active_action(self, tid: str) -> dict | None:
        row = self.db.execute("SELECT id FROM actions WHERE task_id=? AND user_id=? AND status IN (?,?,?,?) ORDER BY created_at LIMIT 1",
                              (tid, self.user_id, ActionStatus.PENDING, ActionStatus.APPROVED,
                               ActionStatus.EXECUTING, ActionStatus.UNKNOWN)).fetchone()
        return self.action(row[0]) if row else None

    def prepare_action(self, tid: str, token: str, tool: str, arguments: dict,
                       fingerprint: str, call: dict, now: float, ttl: float) -> str:
        with self.transaction():
            self.guard(tid, token, now)
            aid = identity("operation")
            expires = float(now + ttl)
            request_hash = digest([self.user_id, tid, tool, arguments, fingerprint, expires])
            self.db.execute("INSERT INTO actions VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL)",
                            (aid, self.user_id, tid, tool, canonical(arguments), fingerprint,
                             canonical(call), ActionStatus.PENDING, now, expires, request_hash))
            self._state(tid, TaskStatus.WAITING_CONFIRMATION, now)
            return aid

    def _action_result(self, action: dict, result: Any, now: float):
        payload = [action["call_json"], {"role": "tool", "tool_call_id": action["id"],
                                        "content": canonical(result)}]
        payload[0] = dict(payload[0], tool_call=dict(payload[0]["tool_call"], id=action["id"]))
        self._message(self.task(action["task_id"]), payload, now)

    def decide(self, aid: str, approve: bool, now: float):
        with self.transaction():
            action = self.action(aid)
            task = self.task(action["task_id"])
            if action["status"] != ActionStatus.PENDING or task["status"] != TaskStatus.WAITING_CONFIRMATION:
                raise AgentError("ACTION_STATE_CONFLICT")
            status = (ActionStatus.EXPIRED if now >= action["expires_at"]
                      else ActionStatus.APPROVED if approve else ActionStatus.REJECTED)
            self.db.execute("UPDATE actions SET status=? WHERE id=? AND user_id=?", (status, aid, self.user_id))
            if status != ActionStatus.APPROVED:
                result = {"ok": False, "error": {"code": "CONFIRMATION_EXPIRED" if status == ActionStatus.EXPIRED else "USER_REJECTED"}}
                self.db.execute("UPDATE actions SET result=? WHERE id=?", (canonical(result), aid))
                self._action_result(action, result, now)
            self._state(task["id"], TaskStatus.READY, now)

    def action_status(self, aid: str, tid: str, token: str, status: ActionStatus, now: float):
        with self.transaction():
            self.guard(tid, token, now)
            self.action(aid)
            self.db.execute("UPDATE actions SET status=? WHERE id=? AND user_id=? AND task_id=?",
                            (status, aid, self.user_id, tid))

    def settle(self, aid: str, tid: str, token: str, result: Any, now: float):
        with self.transaction():
            self.guard(tid, token, now)
            action = self.action(aid)
            if action["task_id"] != tid:
                raise AgentError("NOT_FOUND")
            status = ActionStatus.FAILED if isinstance(result, dict) and result.get("ok") is False else ActionStatus.SUCCEEDED
            self.db.execute("UPDATE actions SET status=?,result=? WHERE id=? AND user_id=?",
                            (status, canonical(result), aid, self.user_id))
            self._action_result(action, result, now)

    def event(self, eid: str, kind: str, payload: Any, now: float):
        if not eid or not kind or len(eid) > 256 or len(kind) > 128 or len(canonical(payload).encode()) > 16384:
            raise AgentError("INVALID_EVENT")
        with self.transaction():
            row = self.db.execute("SELECT kind,payload FROM events WHERE user_id=? AND id=?",
                                  (self.user_id, eid)).fetchone()
            if row:
                if row["kind"] != kind or row["payload"] != canonical(payload):
                    raise AgentError("EVENT_CONFLICT")
                return {"duplicate": True}
            self.db.execute("INSERT INTO events(id,user_id,kind,payload,created_at) VALUES (?,?,?,?,?)",
                            (eid, self.user_id, kind, canonical(payload), now))
            return {"duplicate": False}

    def ready(self, now: float, limit: int) -> list[str]:
        with self.transaction():
            for row in self.db.execute("SELECT id FROM actions WHERE user_id=? AND status=? AND expires_at<=?",
                                       (self.user_id, ActionStatus.PENDING, now)).fetchall():
                self.decide_in_transaction(row["id"], now)
            waiting = self.db.execute("SELECT * FROM tasks WHERE user_id=? AND status=?",
                                      (self.user_id, TaskStatus.WAITING_EVENT)).fetchall()
            for row in waiting:
                event = self.db.execute("SELECT * FROM events WHERE user_id=? AND kind=? AND seq>? ORDER BY seq LIMIT 1",
                                        (self.user_id, row["event_kind"], row["after_event"] or 0)).fetchone() if row["event_kind"] else None
                due = row["wait_until"] is not None and row["wait_until"] <= now
                if event or due:
                    data = {"event_id": event["id"], "kind": event["kind"], "payload": json.loads(event["payload"])} if event else {"wake_at": row["wait_until"]}
                    self._message(dict(row), [{"role": "system", "content": "Wake notification: " + canonical(data)}], now)
                    self._state(row["id"], TaskStatus.READY, now)
            return [row[0] for row in self.db.execute("SELECT id FROM tasks WHERE user_id=? AND (status=? OR (status=? AND lease_until<=?)) ORDER BY created_at,id LIMIT ?",
                                                      (self.user_id, TaskStatus.READY, TaskStatus.RUNNING, now, limit))]

    def decide_in_transaction(self, aid: str, now: float):
        action = self.action(aid)
        if self.task(action["task_id"])["status"] != TaskStatus.WAITING_CONFIRMATION:
            return
        result = {"ok": False, "error": {"code": "CONFIRMATION_EXPIRED"}}
        self.db.execute("UPDATE actions SET status=?,result=? WHERE id=? AND user_id=?",
                        (ActionStatus.EXPIRED, canonical(result), aid, self.user_id))
        self._action_result(action, result, now)
        self._state(action["task_id"], TaskStatus.READY, now)
