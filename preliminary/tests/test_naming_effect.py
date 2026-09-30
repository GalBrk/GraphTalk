"""arm_paths must separate the published-split integer/got files from the
auxiliary corpora that share the same `{model}.*jsonl` glob prefix -- the bug
that made every arm except the two whose extra files just wore off "skipped:
incomplete" (see superseded/docs/paper-revision-handoff.md plan step 4.9, tag `pre-cleanup`)."""

from preliminary.scripts.naming_effect import arm_paths


def test_arm_paths_excludes_auxiliary_corpora(tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  runs = tmp_path / "runs"
  runs.mkdir()
  for name in [
      "qwen3-8b.jsonl", "qwen3-8b.rerun.jsonl",
      "qwen3-8b.shard0of2.jsonl", "qwen3-8b.rerun.shard1of2.jsonl",
      "qwen3-8b.got.jsonl", "qwen3-8b.got.shard0of4.jsonl",
      "qwen3-8b.ladder_screen.jsonl", "qwen3-8b.retrieval_locate.jsonl",
      "qwen3-8b.retrieval.shard0of5.jsonl", "qwen3-8b.retrieval_extend.jsonl",
      "qwen3-8b.retrieval_threshold.shard0of3.jsonl",
      "qwen3-8b.rewire_shared.jsonl", "qwen3-8b.rewire_extra.jsonl",
      "qwen3-8b.size.shard0of2.jsonl", "qwen3-8b.ec500.shard0of5.jsonl",
      "qwen3-8b.cc500.shard0of5.jsonl", "qwen3-8b.probe100.shard0of3.jsonl",
      "qwen3-8b.got.count500.shard0of8.jsonl",
      "qwen3-8b-think.jsonl", "qwen3-8b-think.got.shard0of4.jsonl",
  ]:
    (runs / name).write_text("", encoding="utf-8")

  integer, got = arm_paths(str(runs), "qwen3-8b")

  assert {p.split("\\")[-1].split("/")[-1] for p in integer} == {
      "qwen3-8b.jsonl", "qwen3-8b.rerun.jsonl", "qwen3-8b.shard0of2.jsonl",
      "qwen3-8b.rerun.shard1of2.jsonl"}
  assert {p.split("\\")[-1].split("/")[-1] for p in got} == {
      "qwen3-8b.got.jsonl", "qwen3-8b.got.shard0of4.jsonl"}
