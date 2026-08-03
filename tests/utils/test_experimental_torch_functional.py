import importlib.util
from pathlib import Path
from unittest import mock

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
