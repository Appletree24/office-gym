"""All rows in these unit tests are fabricated fixtures, never experiment evidence."""
import copy
import importlib.util
from pathlib import Path
import sqlite3
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("analysis", ROOT/"scripts"/"analyze_results.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

def fixtures():
    rows = []
    for instance in ("fixture_1", "fixture_2"):
        for group in ("a", "b"):
            for repeat in range(3):
                rows.append({
                    "instance_id":instance, "scenario_family_id":instance, "repeat_id":repeat,
                    "group":group, "split":"id_test", "template_id":"T01",
                    "depth_level":1, "perturbation_level":0,
                    "model_version":"test-model", "reward_version":"test-reward", "harness_version":"test-harness",
                    "env_version":"test-env", "eval_grader_version":"test-grader",
                    "dataset_manifest_hash":"test-manifest", "budget_hash":"test-budget",
                    "success":group=="b", "infra_error":False, "fault_injected":False,
                    "recovery_started":False, "duplicate_effects":0,
                    "premature_completion":False, "declared_done":True,
                    "generated_tokens":10, "input_tokens":20, "tool_calls":2,
                    "wall_seconds":1.0, "sim_seconds":60})
    plan = [{k:r[k] for k in analysis.KEY_FIELDS+analysis.TASK_FIELDS} for r in rows]
    return rows, plan

class StatisticsTests(unittest.TestCase):
    def test_model_revision_drift_within_group_rejected(self):
        rows,plan=fixtures()
        rows[0]['model_version']='changed'
        with self.assertRaisesRegex(ValueError,'Model version changed'):
            analysis.select_and_validate(rows,plan,'a','b','id_test')

    def test_harness_comparison_cannot_change_model(self):
        rows,plan=fixtures()
        for row in rows+plan:
            row['group']='api_simple' if row['group']=='a' else 'api_full'
        for row in rows:
            row['model_version']=row['group']
        with self.assertRaisesRegex(ValueError,'same model version'):
            analysis.select_and_validate(rows,plan,'api_simple','api_full','id_test')

    def test_different_models_allowed_for_training_comparison(self):
        rows,plan=fixtures()
        for row in rows:
            if row['group']=='b':
                row['model_version']='another-checkpoint'
        self.assertEqual(len(analysis.select_and_validate(rows,plan,'a','b','id_test')),12)

    def test_complete_pairing(self):
        rows, plan = fixtures()
        self.assertEqual(len(analysis.select_and_validate(rows,plan,"a","b","id_test")),12)
    def test_missing_result_rejected(self):
        rows, plan = fixtures()
        with self.assertRaises(ValueError):
            analysis.select_and_validate(rows[:-1],plan,"a","b","id_test")
    def test_missing_in_both_groups_detected_by_plan(self):
        rows, plan = fixtures()
        with self.assertRaises(ValueError):
            analysis.select_and_validate([r for r in rows if r["instance_id"]!="fixture_2"],plan,"a","b","id_test")
    def test_incomplete_repeats_rejected(self):
        rows, plan = fixtures()
        rows = [r for r in rows if r["repeat_id"]<2]
        plan = [r for r in plan if r["repeat_id"]<2]
        with self.assertRaises(ValueError):
            analysis.select_and_validate(rows,plan,"a","b","id_test")
    def test_duplicate_rejected(self):
        rows, plan = fixtures()
        with self.assertRaises(ValueError):
            analysis.select_and_validate(rows+[rows[0]],plan,"a","b","id_test")
    def test_inconsistent_version_rejected(self):
        rows, plan = fixtures()
        rows[0]["eval_grader_version"]="different"
        with self.assertRaises(ValueError):
            analysis.select_and_validate(rows,plan,"a","b","id_test")
    def test_known_positive_delta(self):
        rows, _ = fixtures()
        result = analysis.paired_bootstrap(rows,"a","b",samples=200)
        self.assertEqual(result["delta_percentage_points"],100)
        self.assertEqual(result["ci95"]["delta_percentage_points"],[100,100])
    def test_identical_groups_zero_delta(self):
        rows, _ = fixtures()
        for row in rows:
            row["success"]=row["repeat_id"]==0
        result = analysis.paired_bootstrap(rows,"a","b",samples=200)
        self.assertEqual(result["ci95"]["delta_percentage_points"],[0,0])
    def test_family_cluster_count(self):
        rows, _ = fixtures()
        for row in rows:
            row["scenario_family_id"]="shared"
        result=analysis.paired_bootstrap(rows,"a","b","scenario_family_id",samples=200)
        self.assertEqual(result["clusters"],1)
        self.assertIsNone(result["ci95"])
    def test_zero_recovery_denominator_is_null(self):
        rows, _ = fixtures()
        self.assertIsNone(analysis.summary(rows)["recovery_duplicate_episode_rate"])
    def test_fault_denominator(self):
        rows, _ = fixtures()
        rows[0].update(fault_injected=True,recovery_started=True,duplicate_effects=1)
        rows[1].update(fault_injected=True,recovery_started=True)
        self.assertEqual(analysis.summary(rows)["recovery_duplicate_episode_rate"],0.5)
    def test_infra_error_cannot_succeed(self):
        rows, _ = fixtures()
        rows[0].update(success=True,infra_error=True)
        with self.assertRaises(ValueError):
            analysis.validate_row(rows[0])
    def test_synthetic_flag_rejected(self):
        rows, _ = fixtures()
        rows[0]["synthetic"]=True
        with self.assertRaises(ValueError):
            analysis.validate_row(rows[0])
    def test_unseen_has_no_generalization_ci(self):
        rows, _ = fixtures()
        report=analysis.build_report(rows,"a","b","unseen_test",samples=200)
        self.assertIsNone(report["comparison"]["ci95"])

class SQLTests(unittest.TestCase):
    def setUp(self):
        self.connection=sqlite3.connect(":memory:")
        self.connection.executescript((ROOT/"schema"/"world.sql").read_text())
    def tearDown(self):
        self.connection.close()
    def test_event_log_append_only(self):
        self.connection.execute("INSERT INTO event_log(event_id,sim_time,kind,details_json,visibility_json) VALUES ('e1',0,'test','{}','[]')")
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute("DELETE FROM event_log")
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute("UPDATE event_log SET kind='changed'")
    def test_transaction_rolls_back_operation_with_effect(self):
        self.connection.execute("INSERT INTO people VALUES ('p','P','p@example.invalid','UTC','{}')")
        self.connection.commit()
        try:
            with self.connection:
                self.connection.execute("INSERT INTO operations VALUES ('op','p','mail.send','hash','{}',0)")
                self.connection.execute("INSERT INTO event_log(event_id,sim_time,kind,details_json,visibility_json) VALUES ('e1',0,'sent','{}','[]')")
                raise RuntimeError("injected abort before commit")
        except RuntimeError:
            pass
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM operations").fetchone()[0],0)
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM event_log").fetchone()[0],0)

if __name__ == "__main__":
    unittest.main()

