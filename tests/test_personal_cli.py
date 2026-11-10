"""CLI commands preserve state across actual processes without model credentials."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from personal_agent.models import canonical


class CLITests(unittest.TestCase):
    def test_process_restart_and_user_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(canonical({"user_id": "alice", "database": "state.sqlite"}))
            def command(*args, success=True):
                result = subprocess.run([sys.executable, "-m", "personal_agent.cli", "--config", str(config), *args],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0 if success else 1, result.stderr)
                return json.loads(result.stdout if success else result.stderr)
            task = command("chat", "hello", "--no-run")
            self.assertEqual(task["status"], "ready")
            self.assertEqual(command("tasks")[0]["id"], task["id"])
            next_task = command("chat", "another", "--session", task["session_id"], "--no-run")
            self.assertNotEqual(next_task["id"], task["id"])
            self.assertEqual(next_task["session_id"], task["session_id"])
            self.assertEqual(len(command("history", "--session", task["session_id"])), 2)
            self.assertFalse(command("event", "--id", "one", "--kind", "reply")["duplicate"])
            self.assertTrue(command("event", "--id", "one", "--kind", "reply")["duplicate"])
            self.assertEqual(command("cancel", task["id"])["status"], "cancelled")
            self.assertEqual(command("chat", "late", "--task", task["id"], "--no-run", success=False)["error"]["code"], "TASK_STATE_CONFLICT")
            config.write_text(canonical({"user_id": "bob", "database": "state.sqlite"}))
            self.assertEqual(command("tasks"), [])
            self.assertEqual(command("cancel", next_task["id"], success=False)["error"]["code"], "NOT_FOUND")

    def test_missing_model_config_does_not_create_a_task(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(canonical({"user_id": "alice", "database": "state.sqlite"}))
            result = subprocess.run([sys.executable, "-m", "personal_agent.cli", "--config", str(config), "chat", "hello"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(json.loads(result.stderr)["error"]["code"], "MODEL_CONFIG_REQUIRED")
            from personal_agent.store import Store
            store = Store(Path(directory) / "state.sqlite", "alice")
            try:
                self.assertEqual(store.tasks(), [])
            finally:
                store.close()
