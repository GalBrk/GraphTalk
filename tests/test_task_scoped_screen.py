"""`scripts/task_scoped_screen.py` (Phase 1,
`docs/plans/run_improved_tests.md`): the shortcut-ceiling flag and the
`all`-exclusion/pairing logic are new, so they get pinned directly here
rather than trusted only against the real-data regression check the plan
also requires (run separately, against tracked data -- see the module's
own docstring for why "exclude shortcut-ceiling-bound tasks" turned into
"flag shortcut-ceiling-bound (task, condition) pairs" instead)."""

import pandas as pd

from graphtalk import primers
from graphtalk import scoring
from scripts import check_significance as cs
from scripts import task_scoped_screen


def test_shortcut_audit_covers_every_screenable_condition_task_pair():
  """Every (condition, task) pair `screen()` could actually produce --
  every non-`none`, non-derived condition x every task -- must have an
  audit entry, so a future new task/condition can't silently fall through
  to a KeyError at runtime instead of a loud, obvious test failure here."""
  screenable_conditions = [
      c for c in primers.CONDITIONS if c != cs.CONTROL
      and not cs._is_derived_condition(c)
  ]
  missing = [
      (condition, task)
      for condition in screenable_conditions
      for task in scoring.TASKS
      if (condition, task) not in task_scoped_screen._SHORTCUT_AUDIT
  ]
  assert missing == []


def _frame(rows: list[dict]) -> pd.DataFrame:
  """Fills in every column `screen()`/`_paired_values` needs, so each test
  only has to spell out what it's actually varying."""
  base = {
      "is_think": False, "style": "zero_shot", "node_naming": "integer",
  }
  return pd.DataFrame([{**base, **row} for row in rows])


def test_shortcut_flag_looks_up_the_audit_for_a_known_shortcut_pair():
  """degree/edge_count is the audit's canonical exact-theorem shortcut
  (sum of stated degrees / 2) -- see analysis/primer_task_shortcut_audit.md."""
  rows = []
  for i in range(30):
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"edge_count/{i}", "task": "edge_count",
                 "condition": "none", "exact": 0.0, "shortcut_score": 0.018})
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"edge_count/{i}", "task": "edge_count",
                 "condition": "degree", "exact": 1.0, "shortcut_score": 1.0})
  result = task_scoped_screen.screen(_frame(rows), n_perm=200, n_boot=200)
  (row,) = result
  assert row["shortcut_flag"] == "shortcut"
  assert row["shortcut_ceiling_treatment"] == 1.0
  assert row["shortcut_ceiling_control"] == 0.018


def test_shortcut_flag_looks_up_the_audit_for_a_known_none_pair():
  """filler/cycle_check is the audit's canonical "none" pair -- filler
  carries no relational content, ceiling exactly equals the none baseline."""
  rows = []
  for i in range(30):
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"cycle_check/{i}", "task": "cycle_check",
                 "condition": "none", "exact": 0.0, "shortcut_score": 0.832})
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"cycle_check/{i}", "task": "cycle_check",
                 "condition": "filler", "exact": 1.0,
                 "shortcut_score": 0.832})
  result = task_scoped_screen.screen(_frame(rows), n_perm=200, n_boot=200)
  (row,) = result
  assert row["shortcut_flag"] == "none"


def test_shortcut_flag_looks_up_the_audit_for_a_known_partial_pair():
  """clustering/cycle_check is the audit's canonical one-directional
  partial theorem (nonzero clustering anywhere proves a cycle, zero proves
  nothing)."""
  rows = []
  for i in range(30):
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"cycle_check/{i}", "task": "cycle_check",
                 "condition": "none", "exact": 0.0, "shortcut_score": 0.832})
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"cycle_check/{i}", "task": "cycle_check",
                 "condition": "clustering", "exact": 1.0,
                 "shortcut_score": 0.832})
  result = task_scoped_screen.screen(_frame(rows), n_perm=200, n_boot=200)
  (row,) = result
  assert row["shortcut_flag"] == "partial"


def test_all_condition_is_excluded():
  rows = []
  for i in range(30):
    for condition in ("none", "degree", "all"):
      rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                   "instance_id": f"edge_count/{i}", "task": "edge_count",
                   "condition": condition, "exact": 1.0 if condition != "none" else 0.0,
                   "shortcut_score": 1.0})
  result = task_scoped_screen.screen(_frame(rows), n_perm=200, n_boot=200)
  conditions = {r["condition"] for r in result}
  assert conditions == {"degree"}


def test_none_condition_itself_is_excluded_as_a_treatment():
  rows = []
  for i in range(30):
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"edge_count/{i}", "task": "edge_count",
                 "condition": "none", "exact": 0.0, "shortcut_score": 0.018})
  result = task_scoped_screen.screen(_frame(rows), n_perm=200, n_boot=200)
  assert result == []


def test_is_think_rows_are_excluded_from_the_main_sweep_screen():
  rows = []
  for i in range(30):
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"edge_count/{i}", "task": "edge_count",
                 "condition": "none", "exact": 0.0, "shortcut_score": 0.018,
                 "is_think": False})
    rows.append({"model": "qwen3-8b-think", "model_family": "qwen3-8b-think",
                 "instance_id": f"edge_count/{i}", "task": "edge_count",
                 "condition": "degree", "exact": 1.0, "shortcut_score": 1.0,
                 "is_think": True})
  frame = pd.DataFrame([{**{"style": "zero_shot", "node_naming": "integer"}, **r}
                        for r in rows])
  result = task_scoped_screen.screen(frame, n_perm=200, n_boot=200)
  assert result == []  # think-arm rows never paired against a non-think control


def test_keeps_tasks_separate_not_pooled():
  """Two tasks with opposite-sign effects must produce two rows, the same
  per-task discipline `_report_mae`/`_report_exact_per_task` already
  apply, not one pooled number."""
  rows = []
  for i in range(30):
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"edge_count/{i}", "task": "edge_count",
                 "condition": "none", "exact": 0.0, "shortcut_score": 0.018})
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"edge_count/{i}", "task": "edge_count",
                 "condition": "degree", "exact": 1.0, "shortcut_score": 1.0})
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"node_count/{i}", "task": "node_count",
                 "condition": "none", "exact": 1.0, "shortcut_score": 0.064})
    rows.append({"model": "qwen3-8b", "model_family": "qwen3-8b",
                 "instance_id": f"node_count/{i}", "task": "node_count",
                 "condition": "degree", "exact": 0.0, "shortcut_score": 1.0})
  result = {r["task"]: r["delta"] for r in
            task_scoped_screen.screen(_frame(rows), n_perm=200, n_boot=200)}
  assert result["edge_count"] == 1.0
  assert result["node_count"] == -1.0
