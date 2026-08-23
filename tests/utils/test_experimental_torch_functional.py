import importlib.util
from pathlib import Path
from unittest import mock

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[2] / "verl/utils/experimental/torch_functional.py"
_SPEC = importlib.util.spec_from_file_location("experimental_torch_functional", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
FusedLinearForPPO = _MODULE.FusedLinearForPPO


def test_fused_linear_flattens_remove_padding_labels():
    hidden_states = torch.randn(4, 3)
    vocab_weights = torch.randn(5, 3)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    captured = []

    def fake_forward(hidden_states, vocab_weights, input_ids, temperature=1.0):
        captured.append(input_ids.clone())
        token_count = hidden_states.shape[0]
        return torch.zeros(token_count), torch.zeros(token_count)

    with mock.patch.object(_MODULE, "_fused_linear_for_ppo_fwd", side_effect=fake_forward):
        log_probs, entropy = FusedLinearForPPO(chunk_size=2)(
            hidden_states,
            vocab_weights,
            input_ids,
        )

    assert [labels.shape for labels in captured] == [torch.Size([2]), torch.Size([2])]
    assert torch.cat(captured).tolist() == [0, 1, 2, 3]
    assert log_probs.shape == torch.Size([4])
    assert entropy.shape == torch.Size([4])


@pytest.mark.parametrize("contiguous", [True, False], ids=["contiguous", "non_contiguous"])
def test_fused_linear_keeps_hidden_state_grad_path_with_frozen_lm_head(contiguous):
    """Gradient must reach the trainable stack below a 3D hidden state, frozen lm_head included.

    This is the lora shape: peft's "all-linear" leaves lm_head frozen, so the hidden-state
    gradient is the ONLY route back to the adapters. `flatten(0, 1)` inside the autograd
    forward runs with grad mode off, so it preserves requires_grad only while it can return a
    view; on a non-contiguous input it copies and drops the flag, backward then skips
    dhidden_states entirely and returns None, and training silently stops updating the adapter
    while the loss still looks healthy. Both layouts must therefore produce a real gradient.
    """
    torch.manual_seed(0)
    trainable = torch.randn(8, 8, requires_grad=True)
    if contiguous:
        hidden_states = torch.randn(2, 4, 8) @ trainable
    else:
        # transpose without a reshape: a genuinely non-contiguous (bsz, seq, hidden) tensor
        hidden_states = (torch.randn(4, 2, 8) @ trainable).transpose(0, 1)
    assert hidden_states.is_contiguous() is contiguous
    assert hidden_states.ndim == 3

    vocab_weights = torch.randn(16, 8, requires_grad=False)  # lm_head frozen, as lora leaves it
    input_ids = torch.randint(0, 16, (2, 4))

    log_probs, _ = FusedLinearForPPO(chunk_size=2)(hidden_states, vocab_weights, input_ids)
    log_probs.sum().backward()

    assert trainable.grad is not None, "hidden-state gradient was dropped: adapters would never train"
    assert torch.isfinite(trainable.grad).all()
    assert trainable.grad.abs().sum() > 0
