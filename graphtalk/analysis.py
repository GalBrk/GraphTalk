"""A queryable table over the whole tracked sweep, and the failure taxonomy
used to sample cases for manual review.

`scripts/score_sweep.py` scores and prints; nothing before this module
persisted a table, so there was nothing to slice by condition, join against
`analysis/truncated_keys.json`, or sample from. This module is the reusable
layer that does that -- it stays a `graphtalk/` module rather than living
directly in a script because the non-termination heuristic below is exactly
the kind of quiet source of measurement error `graphtalk/scoring.py`'s own
docstring warns about, so it gets the same test-pinned treatment as any other
rule in this package: see `tests/test_analysis.py`, which pins it against
`analysis/truncated_keys.json`'s labelled rows.

Deliberately does not import from `scripts/`: `graphtalk/` is the reusable
layer scripts are built on, not the other way around. Callers (the
`scripts/build_sweep_frame.py` CLI) are expected to load and score records
with `scripts.score_sweep.load`/`score_records` first and pass the already-
scored records in here.
"""

import json
import os
import pathlib

import pandas as pd

from graphtalk import scoring

# Rows that carry a `model` field but are not part of the tracked sweep now live
# in `runs/archive/`, so exclusion is a directory rather than a naming
# convention. That matters: the substrings below were the whole mechanism, and a
# regeneration tagged `redo` would have been dropped from every frame in silence
# -- a live footgun that a directory boundary simply removes.
#
# The substrings are kept as a safety net for paths that predate the move, or
# for anyone globbing the archive back in by hand.
_EXCLUDE_DIR = "archive"
_EXCLUDE_SUBSTRINGS = ("smoke-", ".redo.shard")

# Precedence when a row could be described more than one way: a non-
# terminating row is the more actionable label even when it also happens to
# `parse` (the extractor finds *a* wrong answer in the abandoned working).
FAILURE_TYPES = ("non_terminating", "unparsed", "wrong", "correct")

# Which wording of the prompt a row was generated from. The `filler` primer and
# the `edge_existence` question were both reworded, and only the `zero_shot`
# rows were regenerated -- the obsolete `zero_cot` prompt style keeps the
# original wording, because it is no longer used and not worth the GPU time.
#
# So `condition: filler` does NOT mean one thing across this frame, and pooling
# it across styles averages two different independent variables. That is exactly
# the quiet source of measurement error `graphtalk/scoring.py`'s own docstring
# warns about, which is why this is a column rather than a footnote: group by it,
# or filter on it, but never sum across it.
#
# `unaffected` is the honest label for the 10,800 rows the rewording never
# touched -- for those the two wordings are byte-identical and the distinction
# does not arise.
WORDINGS = ("revised", "original", "unaffected")

# Tukey's extreme-outlier rule (Q3 + 3*IQR), applied per (model, task,
# condition, style) cell rather than as a fixed length cutoff -- there is no
# reliable chars-per-token constant to guess (this project deliberately keeps
# graphtalk/ free of the tokenizer that would give an exact one; see
# graphtalk/models.py), and typical response length varies enormously by task
# and by thinking-vs-not, so a per-cell statistical threshold is the honest
# alternative to a guessed constant.
_OUTLIER_IQR_MULTIPLIER = 3.0


def wording(task: str, condition: str, style: str) -> str:
  """Which prompt wording produced a row; see `WORDINGS`."""
  if condition != "filler" and task != "edge_existence":
    return "unaffected"
  return "revised" if style == "zero_shot" else "original"


def is_excluded(path: str) -> bool:
  """Whether `path` holds rows `docs/DATA.md` says to keep out of the sweep."""
  parts = pathlib.PurePath(path).parts
  if _EXCLUDE_DIR in parts:
    return True
  return any(s in path for s in _EXCLUDE_SUBSTRINGS)


def infer_node_naming(records) -> str:
  """The single `node_naming` scheme every record agrees on, or raises.

  `graphtalk.node_naming.NAMINGS` lists the valid values; absence on a record
  means `"integer"` (`scripts/build_prompts.py`'s convention -- the field is
  only written for a named scheme). Raising on a mix, rather than silently
  keeping whichever scheme happens to be more common, is what lets every
  script downstream skip a `--node-naming` flag entirely: the data says what
  it is, and if it says two different things at once that is exactly the
  collision this exists to catch, before a single row gets mis-scored or
  mis-joined.
  """
  schemes = {r.get("node_naming", "integer") for r in records}
  if len(schemes) > 1:
    raise ValueError(
        f"mixed node_naming schemes in input: {sorted(schemes)} -- score "
        f"each scheme separately, see README.md#node-naming"
    )
  return schemes.pop() if schemes else "integer"


def frame_node_naming(frame: pd.DataFrame) -> str:
  """Like `infer_node_naming`, but for an already-built frame's own column.

  A frame built before this column existed has none at all -- treated as
  `"integer"`, the same absence convention used everywhere else `node_naming`
  is read. Checked independently of `infer_node_naming` (rather than trusting
  a caller upstream already caught a mix), since a frame CSV can come from
  anywhere, not only from `scripts/build_sweep_frame.py`'s own guard.
  """
  if "node_naming" not in frame.columns:
    return "integer"
  schemes = set(frame["node_naming"].unique())
  if len(schemes) > 1:
    raise ValueError(
        f"mixed node_naming schemes in frame: {sorted(schemes)} -- score "
        f"each scheme separately, see README.md#node-naming"
    )
  return next(iter(schemes)) if schemes else "integer"


def tagged_path(path: str, scheme: str) -> str:
  """`path` unchanged for `"integer"`; `.<scheme>` inserted before the
  extension otherwise -- `analysis/sweep_frame.csv` -> `.got.csv`, matching
  the `.rerun.`/`.shard<i>of<n>.` dot-tag convention already live in `runs/`.
  """
  if scheme == "integer":
    return path
  root, ext = os.path.splitext(path)
  return f"{root}.{scheme}{ext}"


def load_truncated_keys(path: str) -> set[tuple[str, str, str, str]]:
  """Ground truth for the non-terminating thinking-arm rows that predate `hit_cap`.

  Originally all 350 known non-terminating rows. Since the 2026-08-29 re-run,
  rows carry their own `hit_cap` and are judged by it, so this file was pruned to
  the 271 rows it still governs -- the ones generated before
  `scripts/run_sweep.py` recorded token counts. It is not a census of
  non-terminating rows any more (that total is 316); it is the fallback for rows
  that cannot state the fact themselves.

  `analysis/truncated_keys.json` is `{model: [[instance_id, condition, style],
  ...]}`; flattened here to `(model, instance_id, condition, style)` tuples
  for O(1) row lookup in `build_frame`.
  """
  with open(path) as handle:
    raw = json.load(handle)
  return {
      (model, instance_id, condition, style)
      for model, keys in raw.items()
      for instance_id, condition, style in keys
  }


def load_shortcuts(path: str) -> dict[tuple[str, str], float]:
  """`shortcuts.json` as a `(task, condition) -> score` dict.

  Same parsing `scripts/score_sweep.py`'s `main` does inline; pulled out here
  so both `scripts/build_sweep_frame.py` and tests can reuse it.
  """
  with open(path) as handle:
    raw = json.load(handle)
  by_cell = {}
  for key, value in raw.items():
    task, condition = key.split("/")
    by_cell[(task, condition)] = value
  return by_cell


def _failure_type(non_terminating: bool, score: dict) -> str:
  if non_terminating:
    return "non_terminating"
  if not score["parsed"]:
    return "unparsed"
  if score["exact"] < 1.0:
    return "wrong"
  return "correct"


def build_frame(
    scored_records: list[dict],
    truncated_keys: set[tuple[str, str, str, str]],
    shortcuts_by_cell: dict[tuple[str, str], float],
) -> pd.DataFrame:
  """The canonical table: one row per scored response.

  `scored_records` is the output of `scripts.score_sweep.score_records` --
  each record already carries `predicted` and `score` (from
  `graphtalk.scoring.extract_answer`/`score_one`) alongside the raw
  `instance_id`/`task`/`condition`/`style`/`gold`/`model`/`response` fields.

  `response` itself is deliberately not a column here: full response text
  runs to ~11K characters on the longest rows, and keeping it out of the
  canonical frame keeps every groupby/export cheap. It is re-joined only in
  `sample_failures`'s companion CLI (`scripts/sample_failures.py`), where full
  text is the actual point.
  """
  rows = []
  for record in scored_records:
    model = record["model"]
    is_think = model.endswith("-think")
    model_family = model[: -len("-think")] if is_think else model
    score = record["score"]
    key = (model, record["instance_id"], record["condition"], record["style"])
    # The row's own `hit_cap` when it has one, the hand-maintained ground-truth
    # file otherwise. Rows generated before `scripts/run_sweep.py` started
    # recording token counts carry no `hit_cap`, and for those
    # `analysis/truncated_keys.json` remains the only record -- so absence must
    # fall through to the lookup rather than read as False.
    recorded_cap = record.get("hit_cap")
    non_terminating = (
        bool(recorded_cap) if recorded_cap is not None else key in truncated_keys
    )
    # Where the flag came from. Since `scripts/backfill_hit_cap.py` ran, every
    # tracked row carries one, so `ground_truth_file` should no longer appear --
    # it stays as the fallback for any row predating both mechanisms.
    cap_source = (
        record.get("token_count_source", "generator")
        if recorded_cap is not None else "ground_truth_file"
    )
    rows.append({
        "instance_id": record["instance_id"],
        "task": record["task"],
        "condition": record["condition"],
        "style": record["style"],
        "gold": record["gold"],
        "model": model,
        "model_family": model_family,
        "is_think": is_think,
        "node_naming": record.get("node_naming", "integer"),
        "predicted": record["predicted"],
        "parsed": score["parsed"],
        "exact": score["exact"],
        "primary": score["primary"],
        "absolute_error": score["absolute_error"],
        "non_terminating": non_terminating,
        "non_terminating_source": cap_source,
        "n_new_tokens": record.get("n_new_tokens"),
        "wording": wording(
            record["task"], record["condition"], record["style"]
        ),
        "has_marker": scoring.has_answer_marker(record["response"]),
        "shortcut_score": shortcuts_by_cell.get(
            (record["task"], record["condition"])
        ),
        "response_len_chars": len(record["response"]),
        "failure_type": _failure_type(non_terminating, score),
    })
  frame = pd.DataFrame(rows)
  if not frame.empty:
    frame["length_outlier"] = _flag_length_outliers(frame)
    # Nullable Int64, not the float64 a column of ints-and-Nones defaults to: a
    # token count is an integer, and `4096.0` in the exported CSV invites the
    # reader to wonder what the fraction means. NA stays NA -- rows generated
    # before the instrumentation have no count, which is not the same as zero.
    frame["n_new_tokens"] = frame["n_new_tokens"].astype("Int64")
  return frame


def _flag_length_outliers(frame: pd.DataFrame) -> pd.Series:
  """Per-(model, task, condition, style) extreme-length flag.

  Unvalidated against any ground truth (unlike `non_terminating`, which comes
  straight from `analysis/truncated_keys.json`) -- this only extends coverage
  to rows the ground-truth file doesn't label: non-`-think` models, or any
  thinking-arm data generated after that file's snapshot. Callers should treat
  it as "worth a human look," not as a second source of truth.
  """
  def threshold(lengths: pd.Series) -> pd.Series:
    q1, q3 = lengths.quantile(0.25), lengths.quantile(0.75)
    iqr = q3 - q1
    return lengths > (q3 + _OUTLIER_IQR_MULTIPLIER * iqr)

  return frame.groupby(
      ["model", "task", "condition", "style"]
  )["response_len_chars"].transform(threshold)


def sample_failures(
    frame: pd.DataFrame, n_per_stratum: int = 3, seed: int = 1234
) -> pd.DataFrame:
  """A stratified sample of non-`correct` rows, for manual inspection.

  Stratifies on `(model, failure_type)` so every model/failure combination
  present gets a look, rather than a plain random sample being dominated by
  whichever failure mode is most common. Fixed `seed` for reproducibility,
  matching this repo's convention of pinned seeds elsewhere (e.g.
  `random_seed=1234` in `scripts/measure_real_rows.py`).
  """
  failures = frame[frame["failure_type"] != "correct"]
  if failures.empty:
    return failures
  # Direct iteration rather than `.groupby(...).apply(...)`: pandas 3.0 made
  # `apply` drop the grouping columns from each sub-frame by default
  # (`include_groups=False`), which would silently strip `model` and
  # `failure_type` from the output. Iterating the groups keeps every column.
  sampled = [
      group.sample(min(len(group), n_per_stratum), random_state=seed)
      for _, group in failures.groupby(["model", "failure_type"], sort=False)
  ]
  return pd.concat(sampled, ignore_index=True)
