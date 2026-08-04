"""Memory-bounded GRPO helpers for selected-token LM-head log-probs.

GRPO does not need the full ``[B, T, V]`` logits tensor; it needs log-probabilities for the sampled
token at each completion position. The primitive here streams the LM head over vocab chunks,
computes ``log_softmax(hidden @ weight.T)[target]``, and recomputes those chunks in backward. That
keeps the public surface useful on CPU and CUDA while leaving room for a later Triton specialization.
"""

from __future__ import annotations

import math

import torch

_DEFAULT_SEQ_CHUNK_SIZE = 2048
_DEFAULT_VOCAB_CHUNK_SIZE = 32768
_SUPPORTED_LOSS_TYPES = frozenset({"grpo", "bnpo", "dr_grpo", "dapo", "luspo"})
_UNIMPLEMENTED_LIGER_LOSS_TYPES = frozenset({"cispo", "sapo", "vespo"})


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _positive_finite_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _check_selective_shapes(hidden: torch.Tensor, weight: torch.Tensor, target_ids: torch.Tensor) -> None:
    if hidden.ndim < 2:
        raise ValueError(f"hidden must have shape [..., H], got {tuple(hidden.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [V, H], got {tuple(weight.shape)}")
    if hidden.shape[-1] != weight.shape[1]:
        raise ValueError(f"hidden dim {hidden.shape[-1]} does not match weight dim {weight.shape[1]}")
    if tuple(hidden.shape[:-1]) != tuple(target_ids.shape):
        raise ValueError(
            f"target_ids shape {tuple(target_ids.shape)} must match hidden leading shape {tuple(hidden.shape[:-1])}"
        )


def _check_same_device(name: str, tensor: torch.Tensor, ref_name: str, ref: torch.Tensor) -> None:
    if tensor.device != ref.device:
        raise ValueError(f"{name} must be on the same device as {ref_name}, got {tensor.device} and {ref.device}")


def _check_integer_dtype(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
        raise ValueError(f"{name} must have integer dtype, got {tensor.dtype}")


def _check_target_ids(target_ids: torch.Tensor, vocab_size: int) -> None:
    if target_ids.numel() == 0:
        return
    if target_ids.device.type != "cpu":
        _ = target_ids.new_empty((vocab_size,), dtype=torch.uint8).index_select(0, target_ids)
        return
    valid = (target_ids >= 0) & (target_ids < vocab_size)
    if bool(valid.all().item()):
        return
    min_id_t, max_id_t = torch.aminmax(target_ids)
    bounds = torch.stack((min_id_t, max_id_t)).cpu()
    min_id = int(bounds[0].item())
    max_id = int(bounds[1].item())
    raise ValueError(f"target_ids must be in [0, {vocab_size}), got min={min_id}, max={max_id}")


def _required_tensor(name: str, tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None:
        raise TypeError(f"{name} is required")
    return tensor


def _check_loss_type(loss_type: str) -> None:
    supported = ", ".join(sorted(_SUPPORTED_LOSS_TYPES))
    if loss_type in _UNIMPLEMENTED_LIGER_LOSS_TYPES:
        raise NotImplementedError(
            f"loss_type='{loss_type}' is a Liger GRPO variant that Chalk does not implement yet; "
            f"supported loss_type values are: {supported}"
        )
    if loss_type not in _SUPPORTED_LOSS_TYPES:
        raise ValueError(f"loss_type must be one of: {supported}")


def _resolve_weight(weight: torch.Tensor | None, lin_weight: torch.Tensor | None) -> torch.Tensor:
    if weight is not None and lin_weight is not None:
        raise TypeError("pass only one of weight or lin_weight")
    if lin_weight is not None:
        return lin_weight
    return _required_tensor("weight", weight)


class _FiniteZeroDependency(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.input_shape = tensor.shape
        ctx.input_dtype = tensor.dtype
        return tensor.new_zeros((), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return torch.zeros(ctx.input_shape, device=grad_output.device, dtype=ctx.input_dtype)


def _cheap_zero_dependency(tensor: torch.Tensor) -> torch.Tensor:
    return _FiniteZeroDependency.apply(tensor)


def _selective_logprob_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: torch.Tensor | None,
    temperature: float,
    seq_chunk_size: int,
    vocab_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for selected-token log-probs, streaming over sequence and vocab chunks."""
    inv_t = 1.0 / float(temperature)
    n_rows = hidden.shape[0]
    vocab_size = weight.shape[0]
    logprobs = torch.empty((n_rows,), device=hidden.device, dtype=torch.float32)
    log_z = torch.empty_like(logprobs)

    for seq_start in range(0, n_rows, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, n_rows)
        h_chunk = hidden[seq_start:seq_end]
        y_chunk = target_ids[seq_start:seq_end]
        rows = seq_end - seq_start

        running_max = torch.full((rows,), float("-inf"), device=hidden.device, dtype=torch.float32)
        running_sum = torch.zeros((rows,), device=hidden.device, dtype=torch.float32)
        target_logit = torch.zeros((rows,), device=hidden.device, dtype=torch.float32)

        for vocab_start in range(0, vocab_size, vocab_chunk_size):
            vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
            w_chunk = weight[vocab_start:vocab_end]
            logits = (h_chunk @ w_chunk.to(h_chunk.dtype).t()).float()
            if bias is not None:
                logits = logits + bias[vocab_start:vocab_end].float()
            logits = logits * inv_t
            local_target_ids = y_chunk - vocab_start
            in_chunk = (local_target_ids >= 0) & (local_target_ids < logits.shape[1])
            safe_local_target_ids = local_target_ids.clamp(0, logits.shape[1] - 1)
            selected_logits = logits.gather(1, safe_local_target_ids.unsqueeze(-1)).squeeze(-1)
            target_logit = torch.where(in_chunk, selected_logits, target_logit)

            chunk_max = logits.amax(dim=-1)
            next_max = torch.maximum(running_max, chunk_max)
            running_sum = running_sum * torch.exp(running_max - next_max) + torch.exp(
                logits - next_max.unsqueeze(-1)
            ).sum(dim=-1)
            running_max = next_max

        lse = running_max + torch.log(running_sum)
        log_z[seq_start:seq_end] = lse
        logprobs[seq_start:seq_end] = target_logit - lse

    return logprobs, log_z


def _selective_logprob_backward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    bias: torch.Tensor | None,
    log_z: torch.Tensor,
    grad_logprobs: torch.Tensor,
    temperature: float,
    seq_chunk_size: int,
    vocab_chunk_size: int,
    *,
    need_grad_hidden: bool = True,
    need_grad_weight: bool = True,
    need_grad_bias: bool = True,
    sanitize_zero_grad_rows: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Backward pass for selected-token log-probs, recomputing logits per vocab chunk."""
    inv_t = 1.0 / float(temperature)
    n_rows = hidden.shape[0]
    vocab_size = weight.shape[0]
    has_bias = bias is not None
    need_grad_bias = bool(has_bias and need_grad_bias)
    if not (need_grad_hidden or need_grad_weight or need_grad_bias):
        return None, None, None

    grad_hidden = torch.zeros_like(hidden, dtype=torch.float32) if need_grad_hidden else None
    if n_rows == 0:
        grad_weight = torch.zeros_like(weight) if need_grad_weight else None
        grad_bias = torch.zeros((vocab_size,), device=weight.device, dtype=torch.float32) if need_grad_bias else None
        return (
            grad_hidden.to(hidden.dtype) if grad_hidden is not None else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
        )

    # Keep the full-size weight gradient in the eventual parameter dtype. Accumulating this buffer
    # in fp32 doubles peak memory for bf16 LM heads and defeats the point of streaming the logits.
    # The first sequence chunk overwrites each vocab slice, avoiding a full zero-fill and a
    # read/modify/write pass over the LM-head-sized gradient buffer.
    grad_weight = torch.empty_like(weight) if need_grad_weight else None
    grad_bias = torch.empty((vocab_size,), device=weight.device, dtype=torch.float32) if need_grad_bias else None
    grad_logprobs = grad_logprobs.float()

    for seq_start in range(0, n_rows, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, n_rows)
        h_chunk = hidden[seq_start:seq_end]
        y_chunk = target_ids[seq_start:seq_end]
        log_z_chunk = log_z[seq_start:seq_end]
        grad_chunk = grad_logprobs[seq_start:seq_end]
        active_grad_rows = grad_chunk != 0 if sanitize_zero_grad_rows else None
        rows = seq_end - seq_start
        row_idx = torch.arange(rows, device=hidden.device)

        for vocab_start in range(0, vocab_size, vocab_chunk_size):
            vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
            w_chunk = weight[vocab_start:vocab_end]
            w_chunk_for_logits = w_chunk.to(h_chunk.dtype)
            logits = (h_chunk @ w_chunk_for_logits.t()).float()
            if has_bias:
                logits = logits + bias[vocab_start:vocab_end].float()
            logits = logits * inv_t

            probs = torch.exp(logits - log_z_chunk.unsqueeze(-1))
            grad_logits = -grad_chunk.unsqueeze(-1) * probs
            if active_grad_rows is not None:
                grad_logits = torch.where(active_grad_rows.unsqueeze(-1), grad_logits, torch.zeros_like(grad_logits))

            in_chunk = (y_chunk >= vocab_start) & (y_chunk < vocab_end)
            local_idx = torch.clamp(y_chunk - vocab_start, 0, vocab_end - vocab_start - 1)
            selected_grad = grad_chunk * in_chunk.float()
            if active_grad_rows is not None:
                selected_grad = torch.where(active_grad_rows, selected_grad, torch.zeros_like(selected_grad))
            grad_logits[row_idx, local_idx] += selected_grad
            grad_logits = grad_logits * inv_t

            grad_matmul = grad_logits.to(h_chunk.dtype)
            if grad_hidden is not None:
                grad_hidden_chunk = (grad_matmul @ w_chunk_for_logits).float()
                if active_grad_rows is not None:
                    grad_hidden_chunk = torch.where(
                        active_grad_rows.unsqueeze(-1),
                        grad_hidden_chunk,
                        torch.zeros_like(grad_hidden_chunk),
                    )
                grad_hidden[seq_start:seq_end] += grad_hidden_chunk
            if grad_weight is not None:
                h_for_grad_weight = h_chunk
                if active_grad_rows is not None:
                    h_for_grad_weight = torch.where(
                        active_grad_rows.unsqueeze(-1),
                        h_chunk,
                        torch.zeros_like(h_chunk),
                    )
                grad_weight_chunk = (grad_matmul.t() @ h_for_grad_weight).to(grad_weight.dtype)
                if seq_start == 0:
                    grad_weight[vocab_start:vocab_end] = grad_weight_chunk
                else:
                    grad_weight[vocab_start:vocab_end] += grad_weight_chunk
            if grad_bias is not None:
                grad_bias_chunk = grad_logits.sum(dim=0)
                if seq_start == 0:
                    grad_bias[vocab_start:vocab_end] = grad_bias_chunk
                else:
                    grad_bias[vocab_start:vocab_end] += grad_bias_chunk

    return (
        grad_hidden.to(hidden.dtype) if grad_hidden is not None else None,
        grad_weight.to(weight.dtype) if grad_weight is not None else None,
        grad_bias.to(bias.dtype) if grad_bias is not None else None,
    )


class _SelectiveLogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden,
        weight,
        target_ids,
        bias,
        temperature,
        seq_chunk_size,
        vocab_chunk_size,
        sanitize_zero_grad_rows,
    ):
        temperature = _positive_finite_float("temperature", temperature)
        logprobs, log_z = _selective_logprob_forward(
            hidden,
            weight,
            target_ids,
            bias,
            temperature,
            seq_chunk_size,
            vocab_chunk_size,
        )
        bias_to_save = hidden.new_empty((0,)) if bias is None else bias
        ctx.save_for_backward(hidden, weight, target_ids, bias_to_save, log_z)
        ctx.has_bias = bias is not None
        ctx.temperature = float(temperature)
        ctx.seq_chunk_size = int(seq_chunk_size)
        ctx.vocab_chunk_size = int(vocab_chunk_size)
        ctx.sanitize_zero_grad_rows = bool(sanitize_zero_grad_rows)
        return logprobs

    @staticmethod
    def backward(ctx, grad_logprobs):
        hidden, weight, target_ids, bias, log_z = ctx.saved_tensors
        need_grad_hidden, need_grad_weight, _, need_grad_bias, _, _, _, _ = ctx.needs_input_grad
        grad_hidden, grad_weight, grad_bias = _selective_logprob_backward(
            hidden,
            weight,
            target_ids,
            bias if ctx.has_bias else None,
            log_z,
            grad_logprobs,
            ctx.temperature,
            ctx.seq_chunk_size,
            ctx.vocab_chunk_size,
            need_grad_hidden=need_grad_hidden,
            need_grad_weight=need_grad_weight,
            need_grad_bias=need_grad_bias,
            sanitize_zero_grad_rows=ctx.sanitize_zero_grad_rows,
        )
        return grad_hidden, grad_weight, None, grad_bias if ctx.has_bias else None, None, None, None, None


def _selective_log_softmax_impl(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    temperature: float = 1.0,
    seq_chunk_size: int = _DEFAULT_SEQ_CHUNK_SIZE,
    vocab_chunk_size: int = _DEFAULT_VOCAB_CHUNK_SIZE,
    sanitize_zero_grad_rows: bool = False,
) -> torch.Tensor:
    _check_selective_shapes(hidden, weight, target_ids)
    _check_same_device("weight", weight, "hidden", hidden)
    _check_same_device("target_ids", target_ids, "hidden", hidden)
    _check_integer_dtype("target_ids", target_ids)
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError(f"bias must have shape [{weight.shape[0]}], got {tuple(bias.shape)}")
    if bias is not None:
        _check_same_device("bias", bias, "hidden", hidden)

    seq_chunk_size = _positive_int("seq_chunk_size", seq_chunk_size)
    vocab_chunk_size = _positive_int("vocab_chunk_size", vocab_chunk_size)
    temperature = _positive_finite_float("temperature", temperature)
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    flat_targets = target_ids.reshape(-1).to(dtype=torch.long).contiguous()
    _check_target_ids(flat_targets, weight.shape[0])
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    out = _SelectiveLogSoftmax.apply(
        flat_hidden,
        weight,
        flat_targets,
        bias,
        float(temperature),
        seq_chunk_size,
        vocab_chunk_size,
        bool(sanitize_zero_grad_rows),
    )
    return out.reshape(target_ids.shape)


def selective_log_softmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    temperature: float = 1.0,
    seq_chunk_size: int = _DEFAULT_SEQ_CHUNK_SIZE,
    vocab_chunk_size: int = _DEFAULT_VOCAB_CHUNK_SIZE,
) -> torch.Tensor:
    """Return ``log_softmax(linear(hidden, weight, bias) / temperature)[target_ids]``.

    The implementation never materializes the full final-dimension logits tensor. ``hidden`` may be
    ``[N, H]`` or ``[B, T, H]``; ``target_ids`` must match the leading dimensions.
    """
    return _selective_log_softmax_impl(
        hidden,
        weight,
        target_ids,
        bias=bias,
        temperature=temperature,
        seq_chunk_size=seq_chunk_size,
        vocab_chunk_size=vocab_chunk_size,
    )


def _reshape_hidden_for_tokens(
    hidden_states: torch.Tensor, selected_token_ids: torch.Tensor, name: str
) -> torch.Tensor:
    if hidden_states.ndim == 3:
        if tuple(hidden_states.shape[:2]) != tuple(selected_token_ids.shape):
            raise ValueError(f"{name} leading dims must match selected_token_ids")
        return hidden_states
    if hidden_states.ndim == 2:
        if hidden_states.shape[0] != selected_token_ids.numel():
            raise ValueError(
                f"{name} flattened token dim must equal selected_token_ids.numel(), "
                f"got {hidden_states.shape[0]} and {selected_token_ids.numel()}"
            )
        return hidden_states.reshape(*selected_token_ids.shape, hidden_states.shape[-1])
    raise ValueError(f"{name} must have shape [B, T, H] or [B*T, H], got {tuple(hidden_states.shape)}")


def _zero_logps_with_grad(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    shape: torch.Size,
) -> torch.Tensor:
    zero = _cheap_zero_dependency(hidden_states) + _cheap_zero_dependency(weight)
    if bias is not None:
        zero = zero + _cheap_zero_dependency(bias)
    return hidden_states.new_zeros(shape, dtype=torch.float32) + zero


def _masked_selective_log_softmax(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    selected_token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    temperature: float,
    seq_chunk_size: int,
    vocab_chunk_size: int,
) -> torch.Tensor:
    _check_selective_shapes(hidden_states, weight, selected_token_ids)
    _check_same_device("selected_token_ids", selected_token_ids, "hidden_states", hidden_states)
    _check_same_device("attention_mask", attention_mask, "hidden_states", hidden_states)
    _check_same_device("weight", weight, "hidden_states", hidden_states)
    _check_integer_dtype("selected_token_ids", selected_token_ids)
    if bias is not None and tuple(bias.shape) != (weight.shape[0],):
        raise ValueError(f"bias must have shape [{weight.shape[0]}], got {tuple(bias.shape)}")
    if bias is not None:
        _check_same_device("bias", bias, "hidden_states", hidden_states)

    temperature = _positive_finite_float("temperature", temperature)
    active_mask = attention_mask > 0
    active_flat = active_mask.reshape(-1)
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_targets = selected_token_ids.to(dtype=torch.long).reshape(-1)
    if not hidden_states.is_cuda:
        active_count = int(active_flat.sum().item())
        if active_count == 0:
            return _zero_logps_with_grad(hidden_states, weight, bias, selected_token_ids.shape)

        if active_count == flat_targets.numel():
            return selective_log_softmax(
                flat_hidden,
                weight,
                flat_targets,
                bias=bias,
                temperature=temperature,
                seq_chunk_size=seq_chunk_size,
                vocab_chunk_size=vocab_chunk_size,
            ).reshape(selected_token_ids.shape)

        active_idx = active_flat.nonzero(as_tuple=False).squeeze(-1)
        active_logps = selective_log_softmax(
            flat_hidden.index_select(0, active_idx),
            weight,
            flat_targets.index_select(0, active_idx),
            bias=bias,
            temperature=temperature,
            seq_chunk_size=seq_chunk_size,
            vocab_chunk_size=vocab_chunk_size,
        )
        flat_out = hidden_states.new_zeros((flat_targets.numel(),), dtype=torch.float32)
        flat_out = flat_out.index_copy(0, active_idx, active_logps.reshape(-1))
        return flat_out.reshape(selected_token_ids.shape)

    safe_hidden = torch.where(active_flat.unsqueeze(-1), flat_hidden, torch.zeros_like(flat_hidden))
    safe_targets = torch.where(active_flat, flat_targets, torch.zeros_like(flat_targets))
    logps = _selective_log_softmax_impl(
        safe_hidden,
        weight,
        safe_targets,
        bias=bias,
        temperature=temperature,
        seq_chunk_size=seq_chunk_size,
        vocab_chunk_size=vocab_chunk_size,
        sanitize_zero_grad_rows=True,
    )
    return torch.where(active_flat, logps, torch.zeros_like(logps)).reshape(selected_token_ids.shape)


def _dapo_token_normalizer(mask: torch.Tensor, num_items_in_batch: torch.Tensor | float | None) -> torch.Tensor:
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()

    if num_items_in_batch is not None:
        normalizer = torch.as_tensor(num_items_in_batch, device=mask.device, dtype=torch.float32).detach()
        if world_size > 1:
            normalizer = normalizer / world_size
        return normalizer.clamp(min=1.0)

    normalizer = mask.float().sum()
    if world_size > 1:
        normalizer = normalizer.clone()
        torch.distributed.all_reduce(normalizer, op=torch.distributed.ReduceOp.SUM)
        normalizer = normalizer / world_size
    return normalizer.clamp(min=1.0)


def grpo_loss_from_logps(
    per_token_logps: torch.Tensor,
    attention_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    ref_per_token_logps: torch.Tensor | None = None,
    old_per_token_logps: torch.Tensor | None = None,
    beta: float = 0.04,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    loss_type: str = "dapo",
    max_completion_length: int | None = None,
    importance_sampling_level: str = "token",
    vllm_is_ratio: torch.Tensor | None = None,
    delta: float | None = None,
    use_bias_correction_kl: bool = False,
    num_items_in_batch: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute the GRPO-family loss from selected-token log-probs.

    Supported reductions are ``grpo``, ``bnpo``, ``dr_grpo``, ``dapo``, and ``luspo``. The function
    covers the common TRL/Liger loss path used by Flash; SAPO/CISPO/VESPO need separate formula
    branches and should be added with downstream coverage.
    """
    _check_loss_type(loss_type)
    if importance_sampling_level not in {"token", "sequence"}:
        raise ValueError("importance_sampling_level must be 'token' or 'sequence'")
    epsilon_low = float(epsilon_low)
    epsilon_high = float(epsilon_high)
    if not math.isfinite(epsilon_low) or epsilon_low < 0:
        raise ValueError(f"epsilon_low must be finite and non-negative, got {epsilon_low}")
    if not math.isfinite(epsilon_high) or epsilon_high < 0:
        raise ValueError(f"epsilon_high must be finite and non-negative, got {epsilon_high}")
    if delta is not None and delta <= 0:
        raise ValueError(f"delta must be positive when set, got {delta}")
    if tuple(per_token_logps.shape) != tuple(attention_mask.shape):
        raise ValueError("per_token_logps and attention_mask must have the same shape")
    if per_token_logps.ndim != 2:
        raise ValueError(f"per_token_logps must have shape [B, T], got {tuple(per_token_logps.shape)}")
    if tuple(advantages.shape) != (per_token_logps.shape[0],):
        raise ValueError(f"advantages must have shape [B], got {tuple(advantages.shape)}")

    raw_logps = per_token_logps.float()
    _check_same_device("attention_mask", attention_mask, "per_token_logps", per_token_logps)
    _check_same_device("advantages", advantages, "per_token_logps", per_token_logps)
    mask = attention_mask.to(dtype=torch.float32)
    mask_bool = mask > 0
    logps = torch.where(mask_bool, raw_logps, torch.zeros_like(raw_logps))
    advantages = advantages.detach().to(dtype=torch.float32)
    if beta != 0.0 and ref_per_token_logps is None:
        raise ValueError("ref_per_token_logps is required when beta != 0")
    if ref_per_token_logps is not None:
        _check_same_device("ref_per_token_logps", ref_per_token_logps, "per_token_logps", per_token_logps)
    if old_per_token_logps is not None:
        _check_same_device("old_per_token_logps", old_per_token_logps, "per_token_logps", per_token_logps)
    ref_logps = logps.detach() if ref_per_token_logps is None else ref_per_token_logps.detach().to(dtype=torch.float32)
    old_logps = logps.detach() if old_per_token_logps is None else old_per_token_logps.detach().to(dtype=torch.float32)

    if tuple(ref_logps.shape) != tuple(logps.shape):
        raise ValueError("ref_per_token_logps must match per_token_logps shape")
    if tuple(old_logps.shape) != tuple(logps.shape):
        raise ValueError("old_per_token_logps must match per_token_logps shape")
    ref_logps = torch.where(mask_bool, ref_logps, torch.zeros_like(ref_logps))
    old_logps = torch.where(mask_bool, old_logps, torch.zeros_like(old_logps))

    log_ratio = logps - old_logps
    if importance_sampling_level == "token":
        log_importance = log_ratio
    else:
        seq_len = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        log_importance = (log_ratio * mask).sum(dim=-1, keepdim=True) / seq_len

    coef = torch.exp(log_importance)
    clipped_coef = torch.clamp(coef, 1.0 - epsilon_low, 1.0 + epsilon_high)
    coef_for_loss = torch.clamp(coef, max=float(delta)) if delta is not None else coef
    adv = advantages.unsqueeze(-1)
    policy_loss = -torch.minimum(coef_for_loss * adv, clipped_coef * adv)
    preserve_luspo_sequence_loss = loss_type == "luspo" and policy_loss.shape != logps.shape

    if vllm_is_ratio is not None:
        _check_same_device("vllm_is_ratio", vllm_is_ratio, "per_token_logps", per_token_logps)
        ratio = vllm_is_ratio.detach().to(dtype=torch.float32)
        if ratio.ndim == 1:
            ratio = ratio.unsqueeze(-1)
        if ratio.shape not in {(logps.shape[0], 1), tuple(logps.shape)}:
            raise ValueError(f"vllm_is_ratio must have shape [B], [B, 1], or [B, T], got {tuple(ratio.shape)}")
        if tuple(ratio.shape) == tuple(logps.shape):
            ratio = torch.where(mask_bool, ratio, torch.ones_like(ratio))
            if preserve_luspo_sequence_loss:
                active_per_row = mask.sum(dim=-1, keepdim=True)
                averaged_ratio = (ratio * mask).sum(dim=-1, keepdim=True) / active_per_row.clamp(min=1.0)
                ratio = torch.where(active_per_row > 0, averaged_ratio, torch.ones_like(averaged_ratio))
        else:
            row_active = mask_bool.any(dim=-1, keepdim=True)
            ratio = torch.where(row_active, ratio, torch.ones_like(ratio))
        policy_loss = policy_loss * ratio

    kl = None
    kl_loss = None
    if beta != 0.0:
        kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1.0
        if use_bias_correction_kl:
            bias_ratio = torch.exp(log_importance)
            if bias_ratio.shape != logps.shape:
                bias_ratio = bias_ratio.expand_as(logps)
            kl = kl * bias_ratio
        kl_loss = float(beta) * kl

    per_token_loss = policy_loss if kl_loss is None else policy_loss + kl_loss

    active = mask.sum().clamp(min=1.0)
    if loss_type == "grpo":
        loss = ((per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)).mean()
    elif loss_type == "bnpo":
        loss = (per_token_loss * mask).sum() / active
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("max_completion_length is required when loss_type='dr_grpo'")
        max_len = _positive_int("max_completion_length", max_completion_length)
        loss = (per_token_loss * mask).sum() / (per_token_logps.shape[0] * max_len)
    elif loss_type == "dapo":
        loss = (per_token_loss * mask).sum() / _dapo_token_normalizer(mask, num_items_in_batch)
    else:  # luspo
        seq_lens = mask.sum(dim=-1, keepdim=True)
        if policy_loss.shape == mask.shape:
            policy_reduced = (policy_loss * mask * seq_lens).mean()
        else:
            policy_reduced = (policy_loss * seq_lens).mean()
        loss = policy_reduced if kl_loss is None else policy_reduced + (kl_loss * mask * seq_lens).mean()

    clip_mask = ((coef < 1.0 - float(epsilon_low)) & (adv < 0)) | ((coef > 1.0 + float(epsilon_high)) & (adv > 0))
    clip_mask = clip_mask.expand_as(mask) if clip_mask.shape != mask.shape else clip_mask
    clip_ratio = (clip_mask.float() * mask).sum() / active
    kl_mean = (kl * mask).sum() / active if kl is not None else None
    metrics = []
    if kl_mean is not None:
        metrics.append(kl_mean.detach())
    metrics.append(clip_ratio.detach())
    return loss, metrics


def fused_linear_grpo_loss(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None = None,
    selected_token_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    *,
    lin_weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    ref_per_token_logps: torch.Tensor | None = None,
    old_per_token_logps: torch.Tensor | None = None,
    ref_input: torch.Tensor | None = None,
    ref_weight: torch.Tensor | None = None,
    ref_bias: torch.Tensor | None = None,
    beta: float = 0.04,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    loss_type: str = "dapo",
    max_completion_length: int | None = None,
    importance_sampling_level: str = "token",
    temperature: float = 1.0,
    use_ref_model: bool = True,
    vllm_is_ratio: torch.Tensor | None = None,
    delta: float | None = None,
    use_bias_correction_kl: bool = False,
    num_items_in_batch: torch.Tensor | float | None = None,
    seq_chunk_size: int = _DEFAULT_SEQ_CHUNK_SIZE,
    vocab_chunk_size: int = _DEFAULT_VOCAB_CHUNK_SIZE,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute GRPO loss without materializing full LM-head logits.

    ``hidden_states`` may be ``[B, T, H]`` or Liger-style ``[B*T, H]``. ``selected_token_ids``
    and ``attention_mask`` are ``[B, T]``,
    matching the Liger fused-linear GRPO loss call shape used by TRL integrations.
    """
    weight = _resolve_weight(weight, lin_weight)
    selected_token_ids = _required_tensor("selected_token_ids", selected_token_ids)
    attention_mask = _required_tensor("attention_mask", attention_mask)
    advantages = _required_tensor("advantages", advantages)
    _check_loss_type(loss_type)
    if tuple(selected_token_ids.shape) != tuple(attention_mask.shape):
        raise ValueError("selected_token_ids and attention_mask must have the same [B, T] shape")
    hidden_states = _reshape_hidden_for_tokens(hidden_states, selected_token_ids, "hidden_states")

    per_token_logps = _masked_selective_log_softmax(
        hidden_states,
        weight,
        selected_token_ids,
        attention_mask,
        bias=bias,
        temperature=temperature,
        seq_chunk_size=seq_chunk_size,
        vocab_chunk_size=vocab_chunk_size,
    )

    if use_ref_model and ref_per_token_logps is None:
        if ref_input is None:
            if beta != 0.0:
                raise ValueError("ref_per_token_logps or ref_input is required when use_ref_model=True and beta != 0")
        else:
            if ref_weight is None:
                raise ValueError("ref_weight is required when ref_input is provided")
            ref_input = _reshape_hidden_for_tokens(ref_input, selected_token_ids, "ref_input")
            ref_per_token_logps = _masked_selective_log_softmax(
                ref_input,
                ref_weight,
                selected_token_ids,
                attention_mask,
                bias=ref_bias,
                temperature=temperature,
                seq_chunk_size=seq_chunk_size,
                vocab_chunk_size=vocab_chunk_size,
            ).detach()

    effective_ref_per_token_logps = ref_per_token_logps if use_ref_model else None
    effective_beta = beta if use_ref_model else 0.0
    return grpo_loss_from_logps(
        per_token_logps,
        attention_mask,
        advantages,
        ref_per_token_logps=effective_ref_per_token_logps,
        old_per_token_logps=old_per_token_logps,
        beta=effective_beta,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
        loss_type=loss_type,
        max_completion_length=max_completion_length,
        importance_sampling_level=importance_sampling_level,
        vllm_is_ratio=vllm_is_ratio,
        delta=delta,
        use_bias_correction_kl=use_bias_correction_kl,
        num_items_in_batch=num_items_in_batch,
    )


class FusedLinearGRPOLoss(torch.nn.Module):
    """Small module wrapper for the Chalk GRPO fused-linear loss."""

    def __init__(
        self,
        beta: float = 0.04,
        compiled: bool = False,
        use_ref_model: bool = True,
        chunk_size: int = 1,
        *,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        loss_type: str = "dapo",
        max_completion_length: int | None = None,
        importance_sampling_level: str = "token",
        temperature: float = 1.0,
        sapo_temperature_pos: float = 1.0,
        sapo_temperature_neg: float = 1.05,
        delta: float | None = None,
        use_bias_correction_kl: bool = False,
        vespo_k_pos: float = 2.0,
        vespo_lambda_pos: float = 3.0,
        vespo_k_neg: float = 3.0,
        vespo_lambda_neg: float = 2.0,
        seq_chunk_size: int = _DEFAULT_SEQ_CHUNK_SIZE,
        vocab_chunk_size: int = _DEFAULT_VOCAB_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        if sapo_temperature_pos <= 0:
            raise ValueError(f"sapo_temperature_pos must be positive, got {sapo_temperature_pos}")
        if sapo_temperature_neg <= 0:
            raise ValueError(f"sapo_temperature_neg must be positive, got {sapo_temperature_neg}")
        if delta is not None and delta <= 0:
            raise ValueError(f"delta must be positive when set, got {delta}")
        _check_loss_type(loss_type)
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.loss_type = loss_type
        self.max_completion_length = max_completion_length
        self.importance_sampling_level = importance_sampling_level
        self.temperature = temperature
        # Compatibility fields for Liger's module API. Chalk's implementation is eager Python and
        # streams over token/vocab chunk sizes below; it does not use torch.compile or Liger's
        # batch-chunk scheduler internally.
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.chunk_size = chunk_size
        self.sapo_temperature_pos = sapo_temperature_pos
        self.sapo_temperature_neg = sapo_temperature_neg
        self.delta = delta
        self.use_bias_correction_kl = use_bias_correction_kl
        self.vespo_k_pos = vespo_k_pos
        self.vespo_lambda_pos = vespo_lambda_pos
        self.vespo_k_neg = vespo_k_neg
        self.vespo_lambda_neg = vespo_lambda_neg
        self.seq_chunk_size = seq_chunk_size
        self.vocab_chunk_size = vocab_chunk_size

    def forward(
        self,
        _input: torch.Tensor,
        weight: torch.Tensor | None = None,
        selected_token_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        ref_per_token_logps: torch.Tensor | None = None,
        old_per_token_logps: torch.Tensor | None = None,
        ref_input: torch.Tensor | None = None,
        ref_weight: torch.Tensor | None = None,
        ref_bias: torch.Tensor | None = None,
        vllm_is_ratio: torch.Tensor | None = None,
        num_items_in_batch: torch.Tensor | float | None = None,
        *,
        lin_weight: torch.Tensor | None = None,
        delta: float | None = None,
        use_bias_correction_kl: bool | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        delta = self.delta if delta is None else delta
        use_bias_correction_kl = (
            self.use_bias_correction_kl if use_bias_correction_kl is None else use_bias_correction_kl
        )
        return fused_linear_grpo_loss(
            _input,
            weight,
            selected_token_ids,
            attention_mask,
            advantages,
            lin_weight=lin_weight,
            bias=bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=self.beta,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            loss_type=self.loss_type,
            max_completion_length=self.max_completion_length,
            importance_sampling_level=self.importance_sampling_level,
            temperature=self.temperature,
            use_ref_model=self.use_ref_model,
            vllm_is_ratio=vllm_is_ratio,
            delta=delta,
            use_bias_correction_kl=use_bias_correction_kl,
            num_items_in_batch=num_items_in_batch,
            seq_chunk_size=self.seq_chunk_size,
            vocab_chunk_size=self.vocab_chunk_size,
        )


__all__ = [
    "FusedLinearGRPOLoss",
    "fused_linear_grpo_loss",
    "grpo_loss_from_logps",
    "selective_log_softmax",
]
