"""Tests for `scripts/naming_effect.py`'s `--tag` filter.

Added for the n=40 density sweep: unlike the tracked ARMS models (each of
which only ever carries one tagged sweep plus its `got` counterpart under
runs/), qwen3-1.7b and qwen3-4b also have runs from several unrelated
experiments in the same directory, so `arm_paths` needs a way to restrict to
one tag's files without pooling the rest in.
"""

from scripts import naming_effect


def _touch(directory, *names):
  for name in names:
    (directory / name).write_text("")
  return directory


def test_arm_paths_without_tag_pools_every_file(tmp_path):
  _touch(tmp_path, "qwen3-1.7b.densfull40.shard0of2.jsonl",
         "qwen3-1.7b.densfull40.got.shard0of2.jsonl",
         "qwen3-1.7b.probe100.shard0of3.jsonl")
  integer, got = naming_effect.arm_paths(str(tmp_path), "qwen3-1.7b")
  assert len(integer) == 2  # densfull40 and probe100, unfiltered
  assert len(got) == 1


def test_arm_paths_tag_restricts_to_matching_files(tmp_path):
  _touch(tmp_path, "qwen3-1.7b.densfull40.shard0of2.jsonl",
         "qwen3-1.7b.densfull40.got.shard0of2.jsonl",
         "qwen3-1.7b.probe100.shard0of3.jsonl")
  integer, got = naming_effect.arm_paths(str(tmp_path), "qwen3-1.7b", tag="densfull40")
  assert integer == [str(tmp_path / "qwen3-1.7b.densfull40.shard0of2.jsonl")]
  assert got == [str(tmp_path / "qwen3-1.7b.densfull40.got.shard0of2.jsonl")]


def test_arm_paths_tag_excludes_unrelated_tags_entirely(tmp_path):
  _touch(tmp_path, "qwen3-1.7b.probe100.shard0of3.jsonl")
  integer, got = naming_effect.arm_paths(str(tmp_path), "qwen3-1.7b", tag="densfull40")
  assert integer == []
  assert got == []
