# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Regression guards for skipping dead old-log-prob and PPO work in direct OPD.

Direct distillation without policy-gradient or task-reward consumers must skip
its old-log-prob forward while preserving the live batch object. Distillation
loss aggregation must also receive the current micro-batch normalization terms
before it runs, even when the task-reward PPO loss is disabled entirely. These
tests exercise those control-flow contracts on CPU with no model or distributed
initialization.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

import verl.trainer.distillation.losses as distillation_losses
import verl.workers.utils.losses as worker_losses
from verl.trainer.main_ppo_sync import PPOTrainer


class _Cfg(dict):
    """A dict that also exposes its keys as attributes, like the OmegaConf the trainer really gets.

    The trainer reads `config.get("distillation")` AND `config.algorithm...` off the same object,
    so a stub has to answer both.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _trainer(
    distillation,
    use_policy_gradient,
    use_task_rewards,
    *,
    rollout_correction=None,
    calculate_log_probs=False,
    use_kl_in_reward=False,
    strategy="fsdp",
    router_replay_mode=None,
):
    actor = _Cfg(strategy=strategy)
    if router_replay_mode is not None:
        actor[strategy] = _Cfg(router_replay=_Cfg(mode=router_replay_mode))
    trainer = SimpleNamespace(
        config=_Cfg(
            distillation=distillation,
            algorithm=_Cfg(
                rollout_correction=rollout_correction,
                use_kl_in_reward=use_kl_in_reward,
            ),
            actor_rollout_ref=_Cfg(
                rollout=_Cfg(calculate_log_probs=calculate_log_probs),
                actor=actor,
            ),
        ),
        distillation_config=SimpleNamespace(
            distillation_loss=SimpleNamespace(
                use_policy_gradient=use_policy_gradient,
                use_task_rewards=use_task_rewards,
            )
        ),
    )
    # the tests call the unbound method with this stub as `self`, so the real helpers have to be
    # bound onto it -- otherwise the delegation resolves to nothing and every case errors rather
    # than exercising the branch under test.
    trainer._router_replay_records_on_this_forward = lambda: PPOTrainer._router_replay_records_on_this_forward(trainer)
    trainer._calculate_log_probs_needs_this_forward = lambda: PPOTrainer._calculate_log_probs_needs_this_forward(
        trainer
    )
    return trainer


@pytest.mark.parametrize(
    ("distillation", "use_policy_gradient", "use_task_rewards", "expected"),
    [
        (None, False, False, True),
        (SimpleNamespace(enabled=True), True, False, True),
        (SimpleNamespace(enabled=True), False, True, True),
        (SimpleNamespace(enabled=True), False, False, False),
    ],
)
def test_old_log_prob_forward_need_truth_table(
    distillation,
    use_policy_gradient,
    use_task_rewards,
    expected,
):
    """A wrong entry either drops an anchor read during actor update or retains a dead GPU forward in direct OPD."""
    assert (
        PPOTrainer._old_log_prob_forward_is_needed(_trainer(distillation, use_policy_gradient, use_task_rewards))
        is expected
    )


@pytest.mark.parametrize(
    ("rollout_correction", "calculate_log_probs", "expected"),
    [
        # the config the skip targets: correction composed by default, but no rollout logprobs
        # exist, so `fit`'s `"rollout_log_probs" in data.batch` term is false and the phase is
        # never admitted. the forward stays skippable.
        ({"bypass_mode": False}, False, False),
        # THE REGRESSION. correction runs and indexes batch["old_log_probs"]
        # (rollout_corr_helper.py:1043) while its own guard only ever checks rollout_log_probs.
        # skipping the forward here KeyErrors before the actor update.
        ({"bypass_mode": False}, True, True),
        # bypass ASSIGNS old_log_probs from rollout_log_probs rather than reading a recomputed
        # one, so it does not consume the forward's output.
        ({"bypass_mode": True}, True, False),
        # correction is off, so nothing READS old_log_probs -- and the forward is still needed.
        # `calculate_log_probs` also gates calculate_debug_metrics inside _compute_old_log_prob,
        # which is an independent need. This expectation was False while the predicate only asked
        # about readers, which is exactly the case that silently dropped requested diagnostics.
        (None, True, True),
    ],
)
def test_rollout_correction_keeps_the_old_log_prob_forward(rollout_correction, calculate_log_probs, expected):
    """Rollout correction reads `old_log_probs` without ever naming it in its own admission test.

    A loss-side-only consumer test looks correct -- direct distillation genuinely ignores the
    proximal anchor -- and still removes a tensor this phase indexes one call earlier.
    """
    trainer = _trainer(
        SimpleNamespace(enabled=True),
        False,
        False,
        rollout_correction=rollout_correction,
        calculate_log_probs=calculate_log_probs,
    )

    assert PPOTrainer._old_log_prob_forward_is_needed(trainer) is expected


@pytest.mark.parametrize("use_kl_in_reward", [False, True])
def test_in_reward_kl_penalty_keeps_the_old_log_prob_forward(use_kl_in_reward):
    """`apply_kl_penalty` indexes old_log_probs unguarded, gated only on `use_kl_in_reward`.

    Same shape as the rollout-correction hazard and independent of every loss-side term: the
    penalty runs in the REWARD phase, so skipping the forward KeyErrors before the actor update
    rather than after it. Nothing in the loss config can be read to infer this one.
    """
    trainer = _trainer(
        SimpleNamespace(enabled=True),
        False,
        False,
        use_kl_in_reward=use_kl_in_reward,
    )

    assert PPOTrainer._old_log_prob_forward_is_needed(trainer) is use_kl_in_reward


@pytest.mark.parametrize(
    ("strategy", "router_replay_mode", "expected"),
    [
        # R2 RECORDS on this forward. `compute_log_prob` carries
        # @_with_routing_replay_flag(enabled=True) and both engines enter RECORD only when
        # forward_only (veomni transformer_impl.py:444, megatron :662). Skipping it leaves
        # `update_actor` -- same decorator, REPLAY branch -- raising "micro_batch missing
        # 'routed_experts'", so the actor update itself breaks.
        ("veomni", "R2", True),
        ("megatron", "R2", True),
        # R3 records on the rollout path instead, so this forward is not its record pass.
        ("veomni", "R3", False),
        ("megatron", "disabled", False),
        # fsdp exposes no router_replay config at all.
        ("fsdp", None, False),
    ],
)
def test_r2_router_replay_keeps_the_old_log_prob_forward(strategy, router_replay_mode, expected):
    """The forward is also a PRODUCER of side effects, not only of `old_log_probs`.

    Nothing in the trainer names router replay -- the coupling is a decorator on the worker method
    plus a forward_only check inside the engine -- so a predicate shaped as "does anything read the
    output" cannot see this one. That is why the predicate asks whether the forward is NEEDED.
    """
    trainer = _trainer(
        SimpleNamespace(enabled=True),
        False,
        False,
        strategy=strategy,
        router_replay_mode=router_replay_mode,
    )

    assert PPOTrainer._old_log_prob_forward_is_needed(trainer) is expected


@pytest.mark.parametrize(
    ("calculate_log_probs", "rollout_correction", "expected"),
    [
        # explicitly requested train/rollout consistency diagnostics: calculate_debug_metrics runs
        # under this flag inside _compute_old_log_prob itself, so skipping the forward silently
        # drops the metrics the run asked for while still paying to collect rollout logprobs.
        (True, None, True),
        (False, None, False),
        # bypass reaches neither need: correction excludes it explicitly and the bypass branch
        # returns before the debug-metrics call.
        (True, {"bypass_mode": True}, False),
    ],
)
def test_requested_log_prob_diagnostics_keep_the_forward(calculate_log_probs, rollout_correction, expected):
    """One flag, two independent needs (correction admission and debug metrics), minus bypass."""
    trainer = _trainer(
        SimpleNamespace(enabled=True),
        False,
        False,
        calculate_log_probs=calculate_log_probs,
        rollout_correction=rollout_correction,
    )

    assert PPOTrainer._old_log_prob_forward_is_needed(trainer) is expected


def test_skipped_old_log_prob_returns_same_batch():
    """A bare return makes the caller rebind batch to None, then a later phase fails while reading batch.extra_info."""
    batch = object()
    trainer = SimpleNamespace(_old_log_prob_forward_is_needed=lambda: False)

    result = PPOTrainer._compute_old_log_prob(trainer, batch, metrics={})

    assert result is batch


def _make_loss_inputs(use_task_rewards: bool):
    # use_kl_loss defaults off: `_diagnostics_without_policy_loss` reads it on every skip, so the
    # stub has to answer it even for tests that only care about the batch-info ordering.
    config = SimpleNamespace(global_batch_info={}, loss_scale_factor=0.25, use_kl_loss=False)
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            use_task_rewards=use_task_rewards,
            distillation_loss_coef=0.5,
        )
    )
    model_output = {"log_probs": torch.tensor([0.0])}
    data = {
        "dp_size": 2,
        "batch_num_tokens": 17,
        "global_batch_size": 8,
    }
    return config, distillation_config, model_output, data


def test_set_global_batch_info_publishes_complete_current_micro_batch():
    """Missing or stale denominators make agg_loss mis-scale a micro-batch or raise before the first actor backward."""
    config, _, _, data = _make_loss_inputs(use_task_rewards=False)

    worker_losses.set_global_batch_info(config, data)

    assert config.global_batch_info == {
        "dp_size": 2,
        "batch_num_tokens": 17,
        "global_batch_size": 8,
        "loss_scale_factor": 0.25,
    }


def test_distillation_ppo_loss_sets_batch_info_before_distillation_aggregation():
    """Publishing after distillation reuses the prior micro-batch denominators, corrupting the loss before backward."""
    config, distillation_config, model_output, data = _make_loss_inputs(use_task_rewards=False)
    observed_batch_info = []

    def record_batch_info(*_args, **_kwargs):
        observed_batch_info.append(dict(config.global_batch_info))
        return torch.tensor(2.0), {}

    with patch.object(distillation_losses, "distillation_loss", side_effect=record_batch_info):
        distillation_losses.distillation_ppo_loss(
            config=config,
            distillation_config=distillation_config,
            model_output=model_output,
            data=data,
        )

    assert observed_batch_info == [
        {
            "dp_size": 2,
            "batch_num_tokens": 17,
            "global_batch_size": 8,
            "loss_scale_factor": 0.25,
        }
    ]


@pytest.mark.parametrize("use_task_rewards", [False, True])
def test_distillation_ppo_loss_calls_ppo_only_for_task_rewards(use_task_rewards):
    """An unconditional call wastes PPO tensor work when disabled.

    An unconditional skip drops task rewards when enabled.
    """
    config, distillation_config, model_output, data = _make_loss_inputs(use_task_rewards=use_task_rewards)
    dp_group = object()

    with (
        patch.object(distillation_losses, "distillation_loss", return_value=(torch.tensor(2.0), {})),
        patch.object(distillation_losses, "ppo_loss", return_value=(torch.tensor(3.0), {})) as ppo_loss,
    ):
        loss, _ = distillation_losses.distillation_ppo_loss(
            config=config,
            distillation_config=distillation_config,
            model_output=model_output,
            data=data,
            dp_group=dp_group,
        )

    if use_task_rewards:
        ppo_loss.assert_called_once_with(config, model_output, data, dp_group)
        torch.testing.assert_close(loss, torch.tensor(4.0))
    else:
        ppo_loss.assert_not_called()
        torch.testing.assert_close(loss, torch.tensor(2.0))


@pytest.mark.parametrize("entropy_present", [False, True])
@pytest.mark.parametrize("use_kl_loss", [False, True])
def test_diagnostics_survive_the_skipped_ppo_loss(entropy_present, use_kl_loss):
    """`ppo_loss` is the ONLY publisher of `actor/entropy_loss`, `kl_loss`, and `kl_coef`.

    Before this branch, `ppo_loss` ran unconditionally and only its SCALAR was zeroed --
    `policy_metrics` survived and was merged. Making the call conditional is what dropped all
    three, so these are regressions rather than pre-existing gaps.

    Both families are configured independently of task rewards (`actor.calculate_entropy` /
    `entropy_coeff`, and `actor.use_kl_loss` via `need_reference_policy`), so the GPU work happens
    either way -- entropy is computed, the reference forward runs -- and only the reporting was
    lost. The negative arms pin the other half: no work done means no metric invented.
    """
    config, distillation_config, model_output, data = _make_loss_inputs(use_task_rewards=False)
    config.loss_agg_mode = "token-mean"
    config.use_kl_loss = use_kl_loss
    config.kl_loss_type = "kl"
    config.kl_loss_coef = 0.001
    model_output["log_probs"] = torch.tensor([[-0.5, -0.25]])
    if entropy_present:
        model_output["entropy"] = torch.tensor([[0.5, 0.25]])
    tensor_data = TensorDict(
        {
            "response_mask": torch.tensor([[1, 1]], dtype=torch.int32),
            "ref_log_prob": torch.tensor([[-0.4, -0.3]]),
        },
        batch_size=[1],
    )
    # the denominators are per-batch scalars, not per-row tensors, so they cannot be set as
    # batched entries on a batch_size=[1] TensorDict.
    for key, value in data.items():
        tensor_data.set_non_tensor(key, value)

    with (
        patch.object(distillation_losses, "distillation_loss", return_value=(torch.tensor(2.0), {})),
        patch.object(distillation_losses, "no_padding_2_padding", side_effect=lambda tensor, _data: tensor),
        patch.object(distillation_losses, "ppo_loss") as ppo_loss,
    ):
        _loss, metrics = distillation_losses.distillation_ppo_loss(
            config=config,
            distillation_config=distillation_config,
            model_output=model_output,
            data=tensor_data,
        )

    ppo_loss.assert_not_called()
    assert ("actor/entropy_loss" in metrics) is entropy_present
    assert ("kl_loss" in metrics) is use_kl_loss
    assert ("kl_coef" in metrics) is use_kl_loss
