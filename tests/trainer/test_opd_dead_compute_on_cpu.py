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
):
    trainer = SimpleNamespace(
        config=_Cfg(
            distillation=distillation,
            algorithm=_Cfg(
                rollout_correction=rollout_correction,
                use_kl_in_reward=use_kl_in_reward,
            ),
            actor_rollout_ref=_Cfg(rollout=_Cfg(calculate_log_probs=calculate_log_probs)),
        ),
        distillation_config=SimpleNamespace(
            distillation_loss=SimpleNamespace(
                use_policy_gradient=use_policy_gradient,
                use_task_rewards=use_task_rewards,
            )
        ),
    )
    # the tests call the unbound method with this stub as `self`, so the real helper has to be
    # bound onto it -- otherwise the delegation resolves to nothing and every case errors rather
    # than exercising the branch under test.
    trainer._rollout_correction_reads_old_log_prob = (
        lambda: PPOTrainer._rollout_correction_reads_old_log_prob(trainer)
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
def test_old_log_prob_consumer_truth_table(
    distillation,
    use_policy_gradient,
    use_task_rewards,
    expected,
):
    """A wrong entry either drops an anchor read during actor update or retains a dead GPU forward in direct OPD."""
    assert PPOTrainer._old_log_prob_has_consumer(
        _trainer(distillation, use_policy_gradient, use_task_rewards)
    ) is expected


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
        # explicitly disabled correction cannot read anything.
        (None, True, False),
    ],
)
def test_rollout_correction_is_counted_as_an_old_log_prob_consumer(
    rollout_correction, calculate_log_probs, expected
):
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

    assert PPOTrainer._old_log_prob_has_consumer(trainer) is expected


@pytest.mark.parametrize("use_kl_in_reward", [False, True])
def test_in_reward_kl_penalty_is_counted_as_an_old_log_prob_consumer(use_kl_in_reward):
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

    assert PPOTrainer._old_log_prob_has_consumer(trainer) is use_kl_in_reward


def test_skipped_old_log_prob_returns_same_batch():
    """A bare return makes the caller rebind batch to None, then a later phase fails while reading batch.extra_info."""
    batch = object()
    trainer = SimpleNamespace(_old_log_prob_has_consumer=lambda: False)

    result = PPOTrainer._compute_old_log_prob(trainer, batch, metrics={})

    assert result is batch


def _make_loss_inputs(use_task_rewards: bool):
    config = SimpleNamespace(global_batch_info={}, loss_scale_factor=0.25)
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
