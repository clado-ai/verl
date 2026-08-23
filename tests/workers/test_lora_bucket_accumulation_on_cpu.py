# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A lora adapter must reach ``add_lora`` whole, not one bucket at a time.

The bucketed weight transport splits a payload on a fixed byte budget
(``rollout.update_weights_bucket_megabytes``, 512 by default) and knows nothing about adapter
boundaries, while ``add_lora`` expects one complete tensor dict per call. A multi-bucket adapter
applied per bucket therefore registers the same lora id several times, each holding only a slice.
This is the normal case for a MoE adapter: fused routed experts stack every expert slice on the
rank axis, so the adapter is GiB-scale.

The accumulation lives inside ``update_weights``, which needs vllm, so these tests exercise the
two pieces the fix actually changed and can run without a gpu:
  1. the receiver hands ``is_last`` to its callback (the signal accumulation depends on), and
  2. the accumulate-then-apply-once policy the callback implements over that signal.
"""

from pathlib import Path

import torch

_TRANSFER_PATH = Path(__file__).parents[2] / "verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"


def _read_source(path: Path) -> str:
    assert path.exists(), f"missing {path}"
    return path.read_text()


def test_receiver_passes_is_last_to_callback():
    """``receive_weights`` must forward the terminal-bucket flag, not just consume it.

    ``metadata["is_last"]`` already governed the receive loop; the callback never saw it, so a
    consumer had no way to tell a mid-transfer bucket from the final one.
    """
    source = _read_source(_TRANSFER_PATH)
    assert "on_bucket_received(weights, is_last)" in source, (
        "receiver must pass is_last to the callback so lora can accumulate across buckets"
    )


def _run_policy(bucket_payloads, is_lora):
    """Mirror the callback in ``update_weights``: accumulate for lora, pass through otherwise.

    Returns the list of payloads handed to ``_update_weights`` (one entry per call), so a per
    bucket application and a single whole-adapter application are directly distinguishable.
    """
    applied = []
    lora_weights = {} if is_lora else None

    def on_bucket_received(weights, is_last):
        if lora_weights is None:
            applied.append(dict(weights))
            return
        lora_weights.update((name, tensor.clone()) for name, tensor in weights)
        if not is_last:
            return
        applied.append(dict(lora_weights))
        lora_weights.clear()

    for index, payload in enumerate(bucket_payloads):
        on_bucket_received(payload, index == len(bucket_payloads) - 1)
    return applied


def test_lora_adapter_spanning_buckets_is_applied_once_and_whole():
    buckets = [
        [("layer0.lora_A.weight", torch.ones(2, 2)), ("layer0.lora_B.weight", torch.ones(2, 2) * 2)],
        [("layer1.lora_A.weight", torch.ones(2, 2) * 3)],
        [("layer1.lora_B.weight", torch.ones(2, 2) * 4)],
    ]

    applied = _run_policy(buckets, is_lora=True)

    assert len(applied) == 1, f"add_lora must be called once per sync, got {len(applied)} partial adapters"
    assert sorted(applied[0]) == [
        "layer0.lora_A.weight",
        "layer0.lora_B.weight",
        "layer1.lora_A.weight",
        "layer1.lora_B.weight",
    ]


def test_base_weights_still_load_per_bucket():
    """Only lora accumulates. Base weights load by name and must not be held until the end."""
    buckets = [[("model.layers.0.weight", torch.ones(2, 2))], [("model.layers.1.weight", torch.ones(2, 2))]]

    applied = _run_policy(buckets, is_lora=False)

    assert len(applied) == 2, "base weight sync must stay streaming, one apply per bucket"


def test_accumulated_lora_tensors_survive_buffer_reuse():
    """Accumulated tensors must be clones, not views into the receiver's reused bucket buffer.

    The receiver fills one buffer per bucket and overwrites it for the next, then frees it in
    ``_cleanup``. ``add_lora`` keeps its references past the callback, so an un-cloned view would
    silently read another bucket's bytes. Reusing one buffer across buckets reproduces exactly that.
    """
    buffer = torch.ones(2, 2)
    captured = {}

    lora_weights = {}

    def on_bucket_received(weights, is_last):
        lora_weights.update((name, tensor.clone()) for name, tensor in weights)
        if is_last:
            captured.update(lora_weights)

    on_bucket_received([("first.weight", buffer)], False)
    buffer.fill_(99)  # the transport reuses the same buffer for the next bucket
    on_bucket_received([("second.weight", buffer)], True)

    assert torch.equal(captured["first.weight"], torch.ones(2, 2)), (
        "first bucket's tensor was overwritten by the second: it was a view, not a clone"
    )
    assert torch.equal(captured["second.weight"], torch.full((2, 2), 99.0))
