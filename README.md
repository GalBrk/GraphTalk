# GraphTalk

Course project building on
[Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560).

`talk_like_a_graph/` is a vendored copy of Google Research's reference
implementation. See [talk_like_a_graph/UPSTREAM.md](talk_like_a_graph/UPSTREAM.md)
for the exact upstream commit and our local changes.

## Setup

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
```

That covers the graph generators, tasks, text encoders, and metrics — which is
everything except `graph_tasks_utils.py`.

### Optional: the TensorFlow pipeline

`graph_tasks_utils.py` alone needs `tensorflow`, `tensorflow-gnn`, and `seqio`
(~2 GB installed):

```bash
uv pip install -e ".[pipeline]"
```

`tensorflow_gnn` 1.0.3 requires Keras 2, but TF 2.20 ships Keras 3, so
`TF_USE_LEGACY_KERAS=1` must be set before importing it. `.venv/bin/activate`
exports it already; set it manually if you run the interpreter without
activating.

Python is pinned to 3.11 because `seqio` and `tensorflow-gnn` do not resolve
cleanly on 3.12+.

## Tests

```bash
uv run --no-sync pytest talk_like_a_graph/ -q
```

27 tests, covering graph generation, text encoders, and metrics.

Use `--no-sync`: a plain `uv run` re-syncs the environment to the default
dependencies and would uninstall the optional `pipeline` extras.
