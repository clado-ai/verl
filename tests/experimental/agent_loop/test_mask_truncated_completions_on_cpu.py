from types import SimpleNamespace
from typing import Any, Optional

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import DictConfigWrap
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Optional[list[dict]] = None,
        add_generation_prompt: bool = True,
        tokenize: bool = True,
        **kwargs,
    ) -> list[int]:
        del messages, tools, add_generation_prompt, tokenize, kwargs
        return [101, 102]


class _FakeServerManager:
    def __init__(self, finish_reason: str):
        self.finish_reason = finish_reason

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> TokenOutput:
        del request_id, prompt_ids, sampling_params, image_data, video_data, audio_data, mm_processor_kwargs
        return TokenOutput(
            token_ids=[11, 12, 13],
            log_probs=[0.0, 0.0, 0.0],
            extra_fields={"finish_reason": self.finish_reason},
        )


class _FakeEngine:
    def __init__(self, finish_reason: str):
        self.finish_reason = finish_reason

    async def generate(self, **kwargs):
        del kwargs
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[11, 12, 13],
                    logprobs=None,
                    routed_experts=None,
                    finish_reason=self.finish_reason,
                )
            ],
            metrics=None,
        )


def _make_agent_loop(mask_truncated_completions: bool) -> SingleTurnAgentLoop:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 16,
                    "response_length": 16,
                    "mask_truncated_completions": mask_truncated_completions,
                    "multi_turn": {"tool_config_path": None},
                },
                "model": {},
            },
            "data": {"tool_config_path": None, "apply_chat_template_kwargs": {}},
        }
    )
    return SingleTurnAgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=_FakeServerManager("stop"),
        tokenizer=_FakeTokenizer(),
        processor=None,
        dataset_cls=RLHFDataset,
        data_config=DictConfigWrap(config.data),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mask_truncated_completions", "finish_reason", "expected_mask"),
    [
        (True, "length", [0, 0, 0]),
        (True, "stop", [1, 1, 1]),
        (False, "length", [1, 1, 1]),
        (False, "stop", [1, 1, 1]),
    ],
)
async def test_single_turn_response_mask(
    mask_truncated_completions: bool,
    finish_reason: str,
    expected_mask: list[int],
):
    agent_loop = _make_agent_loop(mask_truncated_completions)
    agent_loop.server_manager = _FakeServerManager(finish_reason)

    output = await agent_loop.run(sampling_params={}, raw_prompt=[{"role": "user", "content": "hi"}])

    assert output.response_mask == expected_mask


def test_rollout_config_defaults_mask_truncated_completions_off():
    assert RolloutConfig(name="vllm").mask_truncated_completions is False


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "stop"])
async def test_vllm_finish_reason_survives_in_extra_fields(finish_reason: str):
    server = object.__new__(vLLMHttpServer)
    server.config = OmegaConf.create(
        {
            "max_model_len": 16,
            "prompt_length": 8,
            "response_length": 8,
            "repetition_penalty": 1.0,
            "enable_rollout_routing_replay": False,
            "mtp": None,
        }
    )
    server.model_config = SimpleNamespace(processor=None, lora_rank=0, lora={})
    server.engine = _FakeEngine(finish_reason)
    server.global_steps = 7

    output = await server.generate(prompt_ids=[1, 2], sampling_params={}, request_id="request-id")

    assert output.stop_reason == "completed"
    assert output.extra_fields["finish_reason"] == finish_reason
