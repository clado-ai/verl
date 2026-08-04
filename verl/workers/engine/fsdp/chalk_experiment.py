from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CHALK_EXPERIMENT_ENV = "FLASH_CHALK_EXPERIMENT"
CHALK_COMMIT = "a254d8bcd0cb25fd66ca657d226740a5a40d5a34"
CHALK_SOURCE_SHA256 = "8d80f54091a1a02349c5acd579f7da33806554eb72e693e29a79a1c9422f9941"
_REQUIRED_KERNELS = (
    "rmsnorm",
    "fused_lora_delta",
    "trainable_attn_epilogue",
)


def _python_source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_type(module) -> str:
    candidates = [module]
    get_base_model = getattr(module, "get_base_model", None)
    if callable(get_base_model):
        candidates.append(get_base_model())
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        model_type = str(getattr(config, "model_type", ""))
        if model_type:
            return model_type
    return ""


def _validate_report(report: dict) -> None:
    failed = {}
    for name in _REQUIRED_KERNELS:
        result = report.get(name)
        if result is not True:
            failed[name] = result
    forbidden = (
        "swiglu",
        "fused_linear_cross_entropy",
        "attn_epilogue",
        "fp8_frozen_base",
        "fp8_lora_base",
        "fp8_no_wcache",
        "fp8_free_base",
        "fp8_dx",
        "fused_embedding",
        "gdn",
        "moe",
        "rope",
    )
    unexpected = {name: report[name] for name in forbidden if name in report}
    if report.get("liger") is not False:
        unexpected["liger"] = report.get("liger")
    if failed or unexpected:
        raise RuntimeError(
            "chalk experiment did not install the exact required kernel set: "
            + json.dumps({"failed": failed, "unexpected": unexpected}, sort_keys=True, default=repr)
        )


def apply_chalk_experiment(module, *, rank: int) -> dict | None:
    enabled = os.environ.get(CHALK_EXPERIMENT_ENV, "")
    if enabled in ("", "0"):
        return None
    if enabled != "1":
        raise ValueError(f"{CHALK_EXPERIMENT_ENV} must be unset, 0, or 1, got {enabled!r}")

    model_type = _model_type(module)
    if model_type != "qwen3_5":
        raise RuntimeError(f"chalk experiment supports dense qwen3_5 only, got {model_type!r}")

    import chalk
    from chalk.ops import qkv
    from chalk.transformers.apply import apply_chalk_kernel_to_qwen35

    source_root = Path(chalk.__file__).resolve().parent
    source_sha256 = _python_source_digest(source_root)
    if source_sha256 != CHALK_SOURCE_SHA256:
        raise RuntimeError(f"chalk source digest mismatch: expected {CHALK_SOURCE_SHA256}, got {source_sha256}")

    original_out_gate_self_test = qkv.self_test_out_gate
    qkv.self_test_out_gate = lambda: False
    try:
        report = apply_chalk_kernel_to_qwen35(
            module,
            rope=False,
            rmsnorm=True,
            swiglu=False,
            fused_linear_cross_entropy=False,
            fused_lora_delta=True,
            attn_epilogue=False,
            trainable_attn_epilogue=True,
            fused_embedding=False,
            gdn=False,
            moe=False,
            fp8_frozen_base=False,
            fp8_lora_base=False,
            fp8_no_wcache=False,
            fp8_free_base=False,
            fp8_dx=False,
        )
    finally:
        qkv.self_test_out_gate = original_out_gate_self_test
    _validate_report(report)

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

    output_gate_fused = getattr(Qwen3_5Attention, "_chalk_out_gate_fused", None)
    if output_gate_fused is not False:
        raise RuntimeError(f"chalk experiment requires the output gate to remain eager, got {output_gate_fused!r}")

    evidence = {
        "stage": "chalk_install",
        "rank": rank,
        "commit": CHALK_COMMIT,
        "source_sha256": source_sha256,
        "model_type": model_type,
        "required_kernels": list(_REQUIRED_KERNELS),
        "output_gate_fused": output_gate_fused,
        "report": report,
    }
    print(json.dumps(evidence, sort_keys=True, default=repr), flush=True)
    return evidence
