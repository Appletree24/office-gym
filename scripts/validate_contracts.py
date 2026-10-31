#!/usr/bin/env python3
"""Validate bundle contracts; does NOT validate an OfficeGym implementation."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load(path):
    return json.loads((ROOT / path).read_text())

def validate():
    config = load("configs/project.json")
    tools = load("contracts/tools.json")["tools"]
    templates = load("contracts/templates.json")["templates"]
    assert len(tools) == config["counts"]["tools"] == 24
    assert len({tool["name"] for tool in tools}) == 24
    assert len(templates) == config["counts"]["templates"] == 14
    assert len({item["id"] for item in templates}) == 14
    for split, n in config["template_partitions"].items():
        assert sum(t["split"] == split for t in templates) == n
    assert sum(config["counts"][s] for s in ("train","dev","id_test","combo_test","unseen_test")) == config["counts"]["total"]
    ranges = sorted(config["seed_ranges"].values())
    assert all(a <= b for a,b in ranges)
    assert all(left[1] < right[0] for left,right in zip(ranges, ranges[1:]))
    reward = config["reward"]
    assert reward["full_success"] > reward["correct_with_soft_deviation"] > reward["ordinary_failure"] > reward["severe_violation"]
    for script in ("world.sql", "runtime.sql"):
        with sqlite3.connect(":memory:") as connection:
            connection.executescript((ROOT / "schema" / script).read_text())
    for path in ("configs/sft_plan.json", "configs/grpo_plan.json"):
        assert load(path)["compression_allowed"] is False
    schema = load("contracts/run_result.schema.json")
    assert schema["type"] == "object" and "success" in schema["required"]
    print("PASS: 24 tools, 14 templates, planned counts, seed ranges, reward ordering, SQL and training contracts.")
    print("Scope: specification checks only; no MCP, model, environment replay or training was executed.")

if __name__ == "__main__":
    validate()

