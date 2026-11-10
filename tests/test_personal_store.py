"""Persistent assistant invariants use real temporary SQLite files."""
import tempfile
from pathlib import Path
import unittest
from personal_agent.models import AgentError, TaskStatus
from personal_agent.store import Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "state.sqlite"
        self.store = Store(self.path, "alice")
        self.task = self.store.submit("hello", 100)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_user_cannot_read_or_mutate_another_users_state(self):
        sid, tid = self.task["session_id"], self.task["id"]
        source = self.store.history(sid)[0]["id"]
        with self.store.transaction():
            self.store.put_memory("name", "preference", "Alice", [source], 0, 100)
        bob = Store(self.path, "bob")
        try:
            self.assertEqual(bob.tasks(), [])
            self.assertEqual(bob.memory_search(), [])
            for callback in (lambda: bob.task(tid), lambda: bob.history(sid),
                             lambda: bob.cancel(tid, 101),
                             lambda: bob.put_memory("copied", "fact", "x", [source], 0, 101)):
                with self.assertRaises(AgentError):
                    callback()
            self.store.event("same", "reply", {"x": 1}, 100)
            self.assertFalse(bob.event("same", "reply", {"x": 2}, 100)["duplicate"])
        finally:
            bob.close()

    def test_lease_reclaim_rejects_old_executor_after_restart(self):
        tid = self.task["id"]
        old = self.store.claim(tid, 100, 10)
        other = Store(self.path, "alice")
        try:
            self.assertIsNone(other.claim(tid, 109, 10))
            new = other.claim(tid, 110, 10)
            self.assertIsNotNone(new)
            with self.assertRaisesRegex(AgentError, "STALE_LEASE"):
                self.store.transition(tid, old, TaskStatus.COMPLETED, 111)
            other.transition(tid, new, TaskStatus.COMPLETED, 111)
            self.assertEqual(self.store.task(tid)["status"], "completed")
        finally:
            other.close()

    def test_memory_versions_and_user_sources(self):
        tid = self.task["id"]
        token = self.store.claim(tid, 100, 20)
        self.store.append(tid, token, [{"role": "assistant", "content": "guess"}], 100)
        history = self.store.history(self.task["session_id"])
        with self.store.transaction():
            with self.assertRaisesRegex(AgentError, "USER_SOURCE_REQUIRED"):
                self.store.put_memory("name", "preference", "guess", [history[-1]["id"]], 0, 100)
            self.store.put_memory("name", "preference", "Alice", [history[0]["id"]], 0, 100)
            with self.assertRaisesRegex(AgentError, "VERSION_CONFLICT"):
                self.store.put_memory("name", "fact", "Bob", [history[0]["id"]], 0, 101)
            self.store.put_memory("name", "fact", "Alice", [history[0]["id"]], 1, 101)
            with self.assertRaisesRegex(AgentError, "VERSION_CONFLICT"):
                self.store.delete_memory("name", 1)
            self.store.delete_memory("name", 2)
        self.assertEqual(self.store.memory_search(), [])
        with self.store.transaction():
            recreated = self.store.put_memory("name", "fact", "Alice", [history[0]["id"]], 0, 102)
            self.assertEqual(recreated["version"], 4)
            with self.assertRaisesRegex(AgentError, "VERSION_CONFLICT"):
                self.store.put_memory("name", "fact", "stale", [history[0]["id"]], 2, 103)

    def test_wait_only_accepts_events_arriving_after_wait_is_registered(self):
        tid = self.task["id"]
        self.store.event("old", "reply", {"x": 1}, 99)
        token = self.store.claim(tid, 100, 20)
        self.store.transition(tid, token, TaskStatus.WAITING_EVENT, 100, event_kind="reply")
        self.assertEqual(self.store.ready(100, 10), [])
        self.store.close()
        self.store = Store(self.path, "alice")
        self.assertTrue(self.store.event("old", "reply", {"x": 1}, 101)["duplicate"])
        with self.assertRaisesRegex(AgentError, "EVENT_CONFLICT"):
            self.store.event("old", "reply", {"x": 2}, 101)
        self.store.event("new", "reply", {"x": 2}, 101)
        self.assertEqual(self.store.ready(101, 10), [tid])
        self.assertEqual(self.store.history(self.task["session_id"])[-1]["payload"][0]["role"], "system")

    def test_timer_boundary_and_cancelled_task_cannot_be_revived(self):
        tid = self.task["id"]
        token = self.store.claim(tid, 100, 20)
        self.store.transition(tid, token, TaskStatus.WAITING_EVENT, 100, wait_until=110)
        self.assertEqual(self.store.ready(109, 10), [])
        self.assertEqual(self.store.ready(110, 10), [tid])
        token = self.store.claim(tid, 110, 20)
        self.store.cancel(tid, 111)
        with self.assertRaisesRegex(AgentError, "STALE_LEASE"):
            self.store.append(tid, token, [{"role": "assistant", "content": "late"}], 112)
        with self.assertRaisesRegex(AgentError, "TASK_STATE_CONFLICT"):
            self.store.submit("again", 113, task_id=tid)
        self.assertEqual(self.store.ready(200, 10), [])
