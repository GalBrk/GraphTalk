"""Dtype plumbing for hf_backend.load -- the one piece of Titan X support
(no bf16 tensor cores) that is actual logic rather than a doc'd CLI override.

torch/graphtalk.hf_backend are imported inside each test, not at module
scope, matching test_prompts.py's test_run_sweep_row_carries_node_naming --
hf_backend is the only module in this package that requires torch, so
collection must not fail in an env that skipped the pipeline/GPU extras.

Mocks transformers' from_pretrained so this needs no network/GPU: the point
is *which* dtype reaches the loader, not that loading itself works.
"""

from unittest import mock


def _load_with_mocks(**load_kwargs):
    from graphtalk import hf_backend
    from graphtalk import models

    spec = models.ModelSpec(
        "test-spec", "dummy/repo", "test", "0B", "AutoModelForCausalLM", 1
    )
    with mock.patch("transformers.AutoTokenizer.from_pretrained"), \
         mock.patch("transformers.AutoModelForCausalLM.from_pretrained") as from_pretrained:
        from_pretrained.return_value.eval.return_value = from_pretrained.return_value
        hf_backend.load(spec, **load_kwargs)
        return from_pretrained.call_args.kwargs["dtype"]


def test_load_defaults_to_bfloat16():
    import torch
    assert _load_with_mocks() is torch.bfloat16


def test_load_dtype_override_reaches_from_pretrained():
    import torch
    assert _load_with_mocks(dtype="float16") is torch.float16
