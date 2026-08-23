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

These tests drive the real ``vLLMColocateWorkerExtension.update_weights_from_ipc`` and the real
``BucketedWeightReceiver.receive_weights`` loop, with vllm and the device layer stubbed so they run
on cpu. Reimplementing the callback in the test would pass even if production regressed to a
per-bucket apply, so the production functions themselves are what gets called here.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).parents[2]


def _stub_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _RecordingLoRARequest:
    """Stands in for ``TensorLoRARequest`` and keeps what was handed to it."""

    def __init__(self, **kwargs):
        self.lora_tensors = kwargs["lora_tensors"]
        self.peft_config = kwargs["peft_config"]


class _NoopHijack:
    """``vLLMColocateWorkerExtension.__new__`` patches vllm for lora; nothing to patch here."""

    @staticmethod
    def hijack():
        return None


def _fake_torch_device():
    """The device handle the transport calls: ``synchronize`` in the loop, ``ipc_collect`` on cleanup."""
    return types.SimpleNamespace(synchronize=lambda: None, ipc_collect=lambda: None, empty_cache=lambda: None)


def _load_production_modules():
    """Load the two production modules under vllm/device stubs, isolated from real ``verl``.

    Loaded under private names so this file never depends on a vllm install and never mutates the
    importable ``verl`` package for other tests in the session.
    """
    stubs = {
        "vllm": _stub_module("vllm"),
        "vllm.outputs": _stub_module("vllm.outputs", RequestOutput=object),
        "verl.utils.device": _stub_module(
            "verl.utils.device",
            is_npu_available=False,
            get_device_id=lambda: 0,
            get_device_name=lambda: "cpu",
            get_torch_device=_fake_torch_device,
        ),
        "verl.utils.vllm": _stub_module(
            "verl.utils.vllm", TensorLoRARequest=_RecordingLoRARequest, VLLMHijack=_NoopHijack
        ),
        "verl.utils.vllm.patch": _stub_module(
            "verl.utils.vllm.patch", patch_vllm_moe_model_weight_loader=lambda model: None
        ),
        "verl.utils.vllm.vllm_fp8_utils": _stub_module(
            "verl.utils.vllm.vllm_fp8_utils",
            apply_vllm_fp8_patches=lambda *a, **k: None,
            is_fp8_model=lambda *a, **k: False,
            load_quanted_weights=lambda *a, **k: [],
        ),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        loaded = {}
        for alias, relative in (
            ("_lora_bucket_transfer", "verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py"),
            ("_lora_bucket_utils", "verl/workers/rollout/vllm_rollout/utils.py"),
        ):
            spec = importlib.util.spec_from_file_location(alias, _ROOT / relative)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[alias] = module
            spec.loader.exec_module(module)
            loaded[alias] = module
        return loaded["_lora_bucket_transfer"], loaded["_lora_bucket_utils"]
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


_TRANSFER, _UTILS = _load_production_modules()


class _FakeSocket:
    """A ZMQ REP socket that replays a scripted bucket stream.

    ``_init_buffer`` does one ``recv_pyobj`` for the comm handle before the bucket loop, so the
    script starts with that handle and the receiver's own ``send``/``recv`` handshake is exercised.
    """

    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = 0

    def recv_pyobj(self):
        assert self._messages, "receiver asked for more buckets than the script provides"
        return self._messages.pop(0)

    def send(self, _payload):
        self.sent += 1

    def close(self):
        self.closed = True


def _bucket_metadata(entries, is_last, buffer_dtype=torch.float32):
    """Build the wire metadata for one bucket of ``(name, offset, shape)`` entries."""
    return {
        "bucket_meta": {
            name: {"shape": shape, "dtype": buffer_dtype, "offset": offset, "handle": None}
            for name, offset, shape in entries
        },
        "is_last": is_last,
    }


def _scripted_receiver(buffer, script, socket_factory=_FakeSocket):
    """A real ``BucketedWeightReceiver`` whose socket replays ``script`` over ``buffer``.

    ``__new__`` plus explicit attributes rather than ``__init__``: the constructor opens a real ZMQ
    context, and the socket/buffer handshake is what the script stands in for. Everything the test
    exercises -- the receive loop, ``is_last``, cleanup -- is the production implementation.
    """
    receiver = _TRANSFER.BucketedWeightReceiver.__new__(_TRANSFER.BucketedWeightReceiver)
    receiver.device = torch.device("cpu")
    receiver.use_shm = False
    receiver.shm = None
    receiver.buffer = buffer
    receiver.socket = socket_factory(script)
    receiver._init_socket = lambda: None
    receiver._init_buffer = lambda: None
    return receiver


def test_receiver_passes_is_last_to_callback():
    """``receive_weights`` must forward the terminal-bucket flag, not just consume it.

    ``metadata["is_last"]`` already governed the receive loop; the callback never saw it, so a
    consumer had no way to tell a mid-transfer bucket from the final one. Driving the real loop
    proves the flag is passed and is ``True`` on exactly the last bucket.
    """
    buffer = torch.zeros(64, dtype=torch.uint8)
    script = [
        _bucket_metadata([("a", 0, torch.Size([2]))], is_last=False),
        _bucket_metadata([("b", 8, torch.Size([2]))], is_last=False),
        _bucket_metadata([("c", 16, torch.Size([2]))], is_last=True),
    ]
    seen = []

    _scripted_receiver(buffer, script).receive_weights(
        on_bucket_received=lambda weights, is_last: seen.append(([n for n, _ in weights], is_last))
    )

    assert seen == [(["a"], False), (["b"], False), (["c"], True)]


def _make_worker(utils_module, *, peft_config, base_sync_done, buffer, script, socket_factory=_FakeSocket):
    """Build a minimal ``vLLMColocateWorkerExtension`` that runs the real update path on cpu.

    Only the attributes ``update_weights_from_ipc`` actually touches are provided, so the method
    body under test is production code, not a copy.
    """
    worker = utils_module.vLLMColocateWorkerExtension.__new__(utils_module.vLLMColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._is_qat_model = False
    worker._is_modelopt_qat = False
    worker.model_runner = types.SimpleNamespace(vllm_config=None, model=None)
    worker._get_zmq_handle = lambda: "ipc:///unused"
    worker._iter_all_models = lambda: iter(())
    worker._iter_all_models_with_config = lambda: iter(())

    applied = []
    worker.add_lora = lambda request: applied.append(request)
    worker.remove_lora = lambda _lora_id: None
    # base (non-lora) weights go through model.load_weights inside _update_weights; record the
    # per-bucket payloads instead so the streaming path stays observable without a vllm model.
    base_applied = []
    if not (peft_config and base_sync_done):
        worker._update_weights = lambda weights, peft_config, base_sync_done: base_applied.append(
            [name for name, _ in weights]
        )

    # update_weights_from_ipc imports BucketedWeightReceiver from this path at call time, so
    # substituting the module here decides which receiver the production method constructs.
    transfer_module = sys.modules["verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"] = types.ModuleType(
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer"
    )
    transfer_module.BucketedWeightReceiver = lambda zmq_handle, device, use_shm: _scripted_receiver(
        buffer, script, socket_factory
    )
    platforms = sys.modules["vllm.platforms"] = types.ModuleType("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(device_type="cuda")
    return worker, applied, base_applied


@pytest.fixture
def _restore_modules():
    """Stub the modules the production method imports lazily, and undo it afterwards.

    ``update_weights_from_ipc`` imports ``vllm.platforms`` at entry and, on the base-weight path,
    ``vllm.model_executor.model_loader.utils`` after the last bucket. Both are import-time only for
    this test's purposes, so a stub keeps the real method body running without a vllm install.
    """
    names = (
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer",
        "vllm.platforms",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.utils",
    )
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["vllm.model_executor"] = _stub_module("vllm.model_executor")
    sys.modules["vllm.model_executor.model_loader"] = _stub_module("vllm.model_executor.model_loader")
    sys.modules["vllm.model_executor.model_loader.utils"] = _stub_module(
        "vllm.model_executor.model_loader.utils", process_weights_after_loading=lambda *a, **k: None
    )
    yield
    for name, previous in saved.items():
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def test_lora_adapter_spanning_buckets_is_applied_once_and_whole(_restore_modules):
    """The regression itself: a bucket-split adapter must produce exactly one whole ``add_lora``.

    Before the fix, three buckets meant three ``add_lora`` calls, each registering the same lora id
    with a fraction of the tensors -- the last partial registration is what the rollout would serve.
    """
    buffer = torch.zeros(128, dtype=torch.uint8)
    script = [
        _bucket_metadata(
            [("layer0.lora_A.weight", 0, torch.Size([2])), ("layer0.lora_B.weight", 8, torch.Size([2]))], is_last=False
        ),
        _bucket_metadata([("layer1.lora_A.weight", 16, torch.Size([2]))], is_last=False),
        _bucket_metadata([("layer1.lora_B.weight", 24, torch.Size([2]))], is_last=True),
    ]
    worker, applied, _ = _make_worker(_UTILS, peft_config={"r": 8}, base_sync_done=True, buffer=buffer, script=script)

    worker.update_weights_from_ipc(peft_config={"r": 8}, base_sync_done=True)

    assert len(applied) == 1, f"add_lora must be called once per sync, got {len(applied)} partial adapters"
    assert sorted(applied[0].lora_tensors) == [
        "layer0.lora_A.weight",
        "layer0.lora_B.weight",
        "layer1.lora_A.weight",
        "layer1.lora_B.weight",
    ]


def test_base_weights_still_load_per_bucket(_restore_modules):
    """Only lora accumulates. Base weights load by name and must stay streaming, one apply/bucket.

    Holding them to the end would raise peak memory for every non-lora run, which the fix must not do.
    """
    buffer = torch.zeros(64, dtype=torch.uint8)
    script = [
        _bucket_metadata([("model.layers.0.weight", 0, torch.Size([2]))], is_last=False),
        _bucket_metadata([("model.layers.1.weight", 8, torch.Size([2]))], is_last=True),
    ]
    worker, _, base_applied = _make_worker(_UTILS, peft_config=None, base_sync_done=False, buffer=buffer, script=script)

    worker.update_weights_from_ipc(peft_config=None, base_sync_done=False)

    assert base_applied == [["model.layers.0.weight"], ["model.layers.1.weight"]]


def test_accumulated_lora_tensors_survive_buffer_reuse(_restore_modules):
    """Accumulated tensors must not alias the receiver's reused bucket buffer.

    The transport fills one buffer per bucket and overwrites it for the next, then frees it in
    ``_cleanup``, while ``add_lora`` keeps its references well past the callback. Pointing two
    buckets at the same buffer offset and changing the bytes between them reproduces exactly that:
    an un-cloned view would report the second bucket's bytes for the first tensor.
    """
    buffer = torch.zeros(64, dtype=torch.uint8)
    buffer[0:8].view(dtype=torch.float32).fill_(1.0)

    class _MutatingSocket(_FakeSocket):
        def recv_pyobj(self):
            message = super().recv_pyobj()
            # the sender refills the same offset for the next bucket
            if self.sent >= 1:
                buffer[0:8].view(dtype=torch.float32).fill_(99.0)
            return message

    script = [
        _bucket_metadata([("first.weight", 0, torch.Size([2]))], is_last=False),
        _bucket_metadata([("second.weight", 0, torch.Size([2]))], is_last=True),
    ]
    worker, applied, _ = _make_worker(
        _UTILS,
        peft_config={"r": 8},
        base_sync_done=True,
        buffer=buffer,
        script=script,
        socket_factory=_MutatingSocket,
    )

    worker.update_weights_from_ipc(peft_config={"r": 8}, base_sync_done=True)

    assert len(applied) == 1
    tensors = applied[0].lora_tensors
    assert torch.equal(tensors["first.weight"], torch.ones(2)), (
        "first bucket's tensor was overwritten by the second: it was a view, not a clone"
    )
    assert torch.equal(tensors["second.weight"], torch.full((2,), 99.0))


def test_lora_tensors_are_cloned_exactly_once(_restore_modules):
    """One clone, at accumulation. A second one in ``_update_weights`` would double peak memory.

    The accumulated dict is already owned, so re-cloning copies the whole adapter again while the
    first copy is still referenced: a 2x peak on a payload that is GiB-scale for a fused-expert MoE
    adapter (~14.5 GiB at rank 128), at the worst possible moment. Checked by data_ptr identity
    rather than by reading the source, so an equivalent re-copy spelled another way is caught too.
    """
    buffer = torch.zeros(64, dtype=torch.uint8)
    script = [_bucket_metadata([("only.weight", 0, torch.Size([2]))], is_last=True)]
    worker, applied, _ = _make_worker(_UTILS, peft_config={"r": 8}, base_sync_done=True, buffer=buffer, script=script)

    seen_pointers = []
    real_update = worker.__class__._update_weights

    def recording_update(weights, peft_config, base_sync_done):
        seen_pointers.append({name: tensor.data_ptr() for name, tensor in weights})
        return real_update(worker, weights, peft_config=peft_config, base_sync_done=base_sync_done)

    worker._update_weights = recording_update

    worker.update_weights_from_ipc(peft_config={"r": 8}, base_sync_done=True)

    assert len(applied) == 1
    handed_in = seen_pointers[0]["only.weight"]
    stored = applied[0].lora_tensors["only.weight"].data_ptr()
    assert stored == handed_in, (
        "_update_weights re-copied an already-owned tensor: that is a second full copy of the "
        "adapter held alongside the first, i.e. a 2x peak on a GiB-scale payload"
    )
    assert handed_in != buffer.data_ptr(), "accumulated tensor must not alias the receiver buffer"
