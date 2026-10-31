#!/usr/bin/env python3
"""Analyze real OfficeGym runs with a complete run plan and paired cluster bootstrap.
No external dependencies. No example results are included in this package.
"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics

KEY_FIELDS = ("group", "split", "instance_id", "repeat_id")
VERSION_FIELDS = ("env_version", "eval_grader_version", "reward_version", "harness_version", "dataset_manifest_hash", "budget_hash")
TASK_FIELDS = ("scenario_family_id", "template_id", "depth_level", "perturbation_level")
BOOL_FIELDS = ("success", "infra_error", "fault_injected", "recovery_started", "premature_completion", "declared_done")
INT_FIELDS = ("repeat_id", "duplicate_effects", "generated_tokens", "input_tokens", "tool_calls", "sim_seconds")
TEXT_FIELDS = ("instance_id", "scenario_family_id", "group", "split", "template_id") + VERSION_FIELDS + ("model_version",)
ROOT = Path(__file__).resolve().parents[1]

def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return records

def key(row):
    return tuple(row[name] for name in KEY_FIELDS)

def assert_unique(rows, label):
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate run key in {label}")

def validate_row(row):
    schema = json.loads((ROOT / "contracts/run_result.schema.json").read_text())
    missing = set(schema["required"]) - row.keys()
    if missing:
        raise ValueError(f"Missing result fields: {sorted(missing)}")
    if row.get("example_only") or row.get("synthetic"):
        raise ValueError("Example or synthetic data is not accepted for measured reports")
    for name in TEXT_FIELDS:
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError(f"Invalid string: {name}")
    for name in BOOL_FIELDS:
        if type(row[name]) is not bool:
            raise ValueError(f"Expected boolean: {name}")
    for name in INT_FIELDS:
        if type(row[name]) is not int or row[name] < 0:
            raise ValueError(f"Expected nonnegative integer: {name}")
    for name in ("depth_level", "perturbation_level"):
        if type(row[name]) is not int:
            raise ValueError(f"Expected integer: {name}")
    if isinstance(row["wall_seconds"], bool) or not isinstance(row["wall_seconds"], (int, float)) or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0:
        raise ValueError("Invalid wall_seconds")
    if row["success"] and (row["infra_error"] or row["premature_completion"]):
        raise ValueError("Successful run cannot be infrastructure error or premature completion")
    if row["premature_completion"] and not row["declared_done"]:
        raise ValueError("Premature completion requires declared_done")
    if row["recovery_started"] and not row["fault_injected"]:
        raise ValueError("Fault study recovery_started requires fault_injected")

def select_and_validate(rows, plan, group_a, group_b, split, repeats=3):
    if group_a == group_b:
        raise ValueError("Choose two distinct experiment groups")
    selected = [r for r in rows if r.get("group") in (group_a, group_b) and r.get("split") == split]
    planned = [r for r in plan if r.get("group") in (group_a, group_b) and r.get("split") == split]
    if not selected or not planned:
        raise ValueError("No selected results or planned runs")
    for row in selected:
        validate_row(row)
    for row in planned:
        if not all(field in row for field in KEY_FIELDS + TASK_FIELDS):
            raise ValueError("Run plan missing key/task metadata")
        if row.get("example_only") or row.get("synthetic"):
            raise ValueError("Example run plans cannot be used for measured reports")
    assert_unique(selected, "results")
    assert_unique(planned, "plan")
    actual_keys = {key(r) for r in selected}
    planned_keys = {key(r) for r in planned}
    if actual_keys != planned_keys:
        raise ValueError(f"Plan mismatch: missing={len(planned_keys-actual_keys)}, extra={len(actual_keys-planned_keys)}")
    plan_map = {key(r): r for r in planned}
    for row in selected:
        if any(row[f] != plan_map[key(row)][f] for f in TASK_FIELDS):
            raise ValueError("Result task metadata differs from frozen run plan")
    for field in VERSION_FIELDS:
        if len({r[field] for r in selected}) != 1:
            raise ValueError(f"Incomparable versions: {field}")
    groups = {g: [r for r in selected if r["group"] == g] for g in (group_a, group_b)}
    for group,items in groups.items():
        if len({row["model_version"] for row in items})!=1:
            raise ValueError("Model version changed within group: "+group)
    def model_family(group):
        return group.rsplit('_',1)[0] if group.endswith(('_simple','_full')) else group
    if model_family(group_a)==model_family(group_b) and groups[group_a][0]['model_version']!=groups[group_b][0]['model_version']:
        raise ValueError("Harness comparison requires the same model version")
    pair_keys = [{(r["instance_id"], r["repeat_id"]) for r in groups[g]} for g in (group_a, group_b)]
    if pair_keys[0] != pair_keys[1] or not pair_keys[0]:
        raise ValueError("Groups do not contain identical instance/repeat pairs")
    by_instance = defaultdict(list)
    for row in selected:
        by_instance[row["instance_id"]].append(row)
    for instance, items in by_instance.items():
        for field in TASK_FIELDS:
            if len({r[field] for r in items}) != 1:
                raise ValueError(f"Inconsistent {field} for {instance}")
        for group in (group_a, group_b):
            ids = {r["repeat_id"] for r in items if r["group"] == group}
            if ids != set(range(repeats)):
                raise ValueError(f"Expected repeat IDs 0..{repeats-1} for {instance}/{group}")
    return selected

def ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator

def summary(rows):
    n = len(rows)
    success = sum(r["success"] for r in rows)
    recovered = [r for r in rows if r["fault_injected"] and r["recovery_started"]]
    duplicate_runs = sum(r["duplicate_effects"] > 0 for r in recovered)
    premature = sum(r["premature_completion"] for r in rows)
    done = sum(r["declared_done"] for r in rows)
    return {
        "planned_runs": n, "instances": len({r["instance_id"] for r in rows}),
        "success_count": success, "success_rate": ratio(success, n),
        "infra_error_count": sum(r["infra_error"] for r in rows),
        "fault_injected_count": sum(r["fault_injected"] for r in rows),
        "recovery_started_count": len(recovered),
        "recovery_duplicate_run_count": duplicate_runs,
        "recovery_duplicate_episode_rate": ratio(duplicate_runs, len(recovered)),
        "premature_completion_count": premature,
        "premature_completion_rate": ratio(premature, n),
        "premature_given_declared_done": ratio(premature, done),
        "total_generated_tokens": sum(r["generated_tokens"] for r in rows),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_tool_calls": sum(r["tool_calls"] for r in rows),
        "total_wall_seconds": sum(r["wall_seconds"] for r in rows),
        "median_wall_seconds": statistics.median([r["wall_seconds"] for r in rows]) if rows else None
    }

def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lo = int(position)
    hi = min(lo + 1, len(ordered)-1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position-lo)

def paired_bootstrap(rows, group_a, group_b, cluster_field="instance_id", samples=5000, seed=7117):
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    clusters = defaultdict(lambda: {group_a: [], group_b: []})
    for row in rows:
        clusters[row[cluster_field]][row["group"]].append(int(row["success"]))
    totals = []
    for cluster, groups in sorted(clusters.items()):
        a, b = groups[group_a], groups[group_b]
        if not a or len(a) != len(b):
            raise ValueError(f"Unpaired cluster {cluster}")
        totals.append((sum(a), sum(b), len(a)))
    if not totals:
        raise ValueError("No clusters")
    sa = sum(t[0] for t in totals)
    sb = sum(t[1] for t in totals)
    n = sum(t[2] for t in totals)
    point = {"a_success_rate": sa/n, "b_success_rate": sb/n, "delta_percentage_points": 100*(sb-sa)/n}
    if len(totals) < 2:
        return {**point, "clusters": len(totals), "ci95": None, "warning": "Fewer than two independent clusters"}
    rng = random.Random(seed)
    draws_a, draws_b, deltas = [], [], []
    for _ in range(samples):
        chosen = [totals[rng.randrange(len(totals))] for _ in totals]
        denom = sum(t[2] for t in chosen)
        pa = sum(t[0] for t in chosen)/denom
        pb = sum(t[1] for t in chosen)/denom
        draws_a.append(pa)
        draws_b.append(pb)
        deltas.append(100*(pb-pa))
    def ci(x):
        return [quantile(x, 0.025), quantile(x, 0.975)]
    return {**point, "clusters": len(totals), "cluster_field": cluster_field,
            "ci95": {"a_success_rate": ci(draws_a), "b_success_rate": ci(draws_b),
                     "delta_percentage_points": ci(deltas)},
            "bootstrap_samples": samples, "bootstrap_seed": seed}

def build_report(rows, group_a, group_b, split, cluster_field="instance_id", samples=5000):
    report = {"split": split, "group_a": group_a, "group_b": group_b,
              "versions": {k: rows[0][k] for k in VERSION_FIELDS},
              "model_versions": {g: sorted({r["model_version"] for r in rows if r["group"]==g}) for g in (group_a,group_b)},
              "groups": {g: summary([r for r in rows if r["group"] == g]) for g in (group_a, group_b)},
              "limitations": [
                  "Evaluation repeats do not capture training-seed uncertainty.",
                  "Equal limits do not imply equal actual compute or API cost.",
                  "A bundled Harness comparison does not identify individual component effects.",
                  "Freeze and hash the run plan before execution; this script cannot verify its creation time."
              ]}
    if split == "unseen_test":
        report["comparison"] = {"ci95": None, "reason": "Few held-out templates: report per template, no cross-template generalization CI"}
    else:
        report["comparison"] = paired_bootstrap(rows, group_a, group_b, cluster_field, samples)
    report["per_template"] = {
        template: {g: summary([r for r in rows if r["template_id"] == template and r["group"] == g]) for g in (group_a, group_b)}
        for template in sorted({r["template_id"] for r in rows})
    }
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True, help="Frozen planned-run JSONL; never derive it from successful results")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--split", choices=("dev","id_test","combo_test","unseen_test"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cluster", choices=("instance_id","scenario_family_id"), default="instance_id")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    rows = select_and_validate(read_jsonl(args.input), read_jsonl(args.plan), args.a, args.b, args.split, args.repeats)
    if args.cluster == "instance_id":
        families = defaultdict(set)
        for row in rows:
            families[row["scenario_family_id"]].add(row["instance_id"])
        if any(len(ids)>1 for ids in families.values()):
            raise ValueError("Related variants detected: use --cluster scenario_family_id")
    report = build_report(rows, args.a, args.b, args.split, args.cluster, args.bootstrap_samples)
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / f"{args.split}__{args.a}__{args.b}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    csv_path = report_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group","depth_level","perturbation_level","runs","success_rate"])
        writer.writeheader()
        for group in (args.a, args.b):
            pairs = sorted({(r["depth_level"],r["perturbation_level"]) for r in rows})
            for depth, perturbation in pairs:
                items = [r for r in rows if r["group"]==group and r["depth_level"]==depth and r["perturbation_level"]==perturbation]
                if items:
                    s = summary(items)
                    writer.writerow({"group":group,"depth_level":depth,"perturbation_level":perturbation,
                                     "runs":len(items),"success_rate":s["success_rate"]})
    print(f"Report: {report_path}; stratified CSV: {csv_path}")
    print("CSV cells are descriptive. Hold other factors fixed before drawing a depth/perturbation curve.")

if __name__ == "__main__":
    main()

