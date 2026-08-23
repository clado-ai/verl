import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[2] / "verl/models/transformers/qwen3_5.py"


def _load_qwen35_module():
    verl_module = types.ModuleType("verl")
    utils_module = types.ModuleType("verl.utils")
    ulysses_module = types.ModuleType("verl.utils.ulysses")
    ulysses_module.get_ulysses_sequence_parallel_world_size = lambda: 1
    ulysses_module.ulysses_pad_and_slice_inputs = lambda labels, **_kwargs: (labels, None, 0)

    # Import torch's lazy subsystems BEFORE the patch. `mock.patch.dict` restores the exact
    # sys.modules snapshot it took, so anything imported inside the block is EVICTED on exit --
    # including the ~90 `torch._dynamo` modules a first autograd/compile touch pulls in. The next
    # test to import them re-executes `torch._inductor.test_operators`, which re-runs a
    # TORCH_LIBRARY registration that may only happen once per process, and dies with
    # "Only a single TORCH_LIBRARY can be used to register the namespace _inductor_test".
    # Importing here pins them in the snapshot, so the restore keeps rather than drops them.
    import torch._dynamo  # noqa: F401

    with mock.patch.dict(
        sys.modules,
        {
            "verl": verl_module,
            "verl.utils": utils_module,
            "verl.utils.ulysses": ulysses_module,
        },
    ):
        spec = importlib.util.spec_from_file_location("isolated_qwen3_5", _MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


_QWEN35 = _load_qwen35_module()


class _FakeBaseOutput:
    def __init__(self, hidden_states):
        self._hidden_states = hidden_states
        self.hidden_states = None

    def __getitem__(self, index):
        if index == 0:
            return self._hidden_states
        raise IndexError(index)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(8, 64, bias=False)

    def model(self, input_ids, **_kwargs):
        hidden_states = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
        return _FakeBaseOutput(hidden_states)


def _run_forward(forward_fn, *, shift_labels):
    captured = {}

    class FakeFusedLinearForPPO:
        def forward(self, hidden_states, vocab_weights, input_ids, temperature=1.0):
            captured["input_ids"] = input_ids.detach().clone()
            shape = input_ids.shape
            return torch.zeros(shape), torch.zeros(shape)

    def fake_linear_cross_entropy(hidden_states, vocab_weights, input_ids, temperature, reduction):
        captured["input_ids"] = input_ids.detach().clone()
        shape = input_ids.shape
        return torch.zeros(shape), torch.zeros(shape)

    experimental_module = types.ModuleType("verl.utils.experimental")
    torch_functional_module = types.ModuleType("verl.utils.experimental.torch_functional")
    torch_functional_module.FusedLinearForPPO = FakeFusedLinearForPPO
    kernel_module = types.ModuleType("verl.utils.kernel")
    linear_ce_module = types.ModuleType("verl.utils.kernel.linear_cross_entropy")
    linear_ce_module.linear_cross_entropy = fake_linear_cross_entropy

    slice_calls = []

    def fake_slice(labels, **_kwargs):
        slice_calls.append(labels.detach().clone())
        return labels[..., : labels.shape[-1] // 2], None, 0

    input_ids = torch.tensor([[10, 20, 30, 40]])
    with mock.patch.dict(
        sys.modules,
        {
            "verl.utils.experimental": experimental_module,
            "verl.utils.experimental.torch_functional": torch_functional_module,
            "verl.utils.kernel": kernel_module,
            "verl.utils.kernel.linear_cross_entropy": linear_ce_module,
        },
    ):
        with (
            mock.patch.object(_QWEN35, "get_ulysses_sequence_parallel_world_size", return_value=2),
            mock.patch.object(_QWEN35, "ulysses_pad_and_slice_inputs", side_effect=fake_slice),
        ):
            forward_fn(
                _FakeModel(),
                input_ids=input_ids,
                shift_labels=shift_labels,
                temperature=1.0,
            )

    return captured["input_ids"], slice_calls


@pytest.mark.parametrize(
    "forward_fn",
    [_QWEN35.forward_with_torch_backend, _QWEN35.forward_with_triton_backend],
)
def test_engine_shift_labels_are_not_sliced_twice(forward_fn):
    shift_labels = torch.tensor([[20, 30, 40, 50]])

    captured_labels, slice_calls = _run_forward(forward_fn, shift_labels=shift_labels)

    assert slice_calls == []
    assert torch.equal(captured_labels, shift_labels)


@pytest.mark.parametrize(
    "forward_fn",
    [_QWEN35.forward_with_torch_backend, _QWEN35.forward_with_triton_backend],
)
def test_local_labels_keep_ulysses_fallback_slice(forward_fn):
    captured_labels, slice_calls = _run_forward(forward_fn, shift_labels=None)

    assert len(slice_calls) == 1
    assert captured_labels.shape == torch.Size([1, 2])
