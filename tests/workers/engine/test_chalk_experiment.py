from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def chalk_experiment_module():
    root = Path(__file__).parents[3]
    path = root / "verl/workers/engine/fsdp/chalk_experiment.py"
    spec = importlib.util.spec_from_file_location("chalk_experiment_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_chalk(monkeypatch, tmp_path, installer):
    package = tmp_path / "chalk"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("value = 1\n")

    chalk = types.ModuleType("chalk")
    chalk.__file__ = str(init)
    chalk_transformers = types.ModuleType("chalk.transformers")
    chalk_transformers.__path__ = []
    apply = types.ModuleType("chalk.transformers.apply")
    apply.apply_chalk_kernel_to_qwen35 = installer
    ops = types.ModuleType("chalk.ops")
    ops.__path__ = []
    qkv = types.ModuleType("chalk.ops.qkv")
    qkv.self_test_out_gate = lambda: True

    transformers = types.ModuleType("transformers")
    transformers.__path__ = []
    models = types.ModuleType("transformers.models")
    models.__path__ = []
    qwen = types.ModuleType("transformers.models.qwen3_5")
    qwen.__path__ = []
    modeling = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    modeling.Qwen3_5Attention = type("Qwen3_5Attention", (), {"_chalk_out_gate_fused": False})

    monkeypatch.setitem(sys.modules, "chalk", chalk)
    monkeypatch.setitem(sys.modules, "chalk.ops", ops)
    monkeypatch.setitem(sys.modules, "chalk.ops.qkv", qkv)
    monkeypatch.setitem(sys.modules, "chalk.transformers", chalk_transformers)
    monkeypatch.setitem(sys.modules, "chalk.transformers.apply", apply)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.models", models)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5", qwen)
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5", modeling)
    return package


def _model(model_type="qwen3_5"):
    return SimpleNamespace(config=SimpleNamespace(model_type=model_type))


def test_model_type_resolves_peft_base_model(chalk_experiment_module):
    wrapped = SimpleNamespace(config=SimpleNamespace(), get_base_model=lambda: _model())

    assert chalk_experiment_module._model_type(wrapped) == "qwen3_5"


def _passing_report():
    return {
        "liger": False,
        "rmsnorm": True,
        "fused_lora_delta": True,
        "trainable_attn_epilogue": True,
    }


def test_experiment_is_default_off_without_importing_chalk(monkeypatch, chalk_experiment_module):
    monkeypatch.delenv(chalk_experiment_module.CHALK_EXPERIMENT_ENV, raising=False)
    monkeypatch.delitem(sys.modules, "chalk", raising=False)

    assert chalk_experiment_module.apply_chalk_experiment(_model(), rank=0) is None
    assert "chalk" not in sys.modules


def test_experiment_applies_only_required_kernel_set(monkeypatch, tmp_path, chalk_experiment_module):
    captured = {}

    def installer(module, **kwargs):
        captured["module"] = module
        captured["kwargs"] = kwargs
        captured["output_gate_during_install"] = sys.modules["chalk.ops.qkv"].self_test_out_gate()
        return _passing_report()

    package = _fake_chalk(monkeypatch, tmp_path, installer)
    monkeypatch.setattr(
        chalk_experiment_module,
        "CHALK_SOURCE_SHA256",
        chalk_experiment_module._python_source_digest(package),
    )
    monkeypatch.setenv(chalk_experiment_module.CHALK_EXPERIMENT_ENV, "1")
    model = _model()

    evidence = chalk_experiment_module.apply_chalk_experiment(model, rank=3)

    assert captured["module"] is model
    assert captured["kwargs"] == {
        "rope": False,
        "rmsnorm": True,
        "swiglu": False,
        "fused_linear_cross_entropy": False,
        "fused_lora_delta": True,
        "attn_epilogue": False,
        "trainable_attn_epilogue": True,
        "fused_embedding": False,
        "gdn": False,
        "moe": False,
        "fp8_frozen_base": False,
        "fp8_lora_base": False,
        "fp8_no_wcache": False,
        "fp8_free_base": False,
        "fp8_dx": False,
    }
    assert captured["output_gate_during_install"] is False
    assert sys.modules["chalk.ops.qkv"].self_test_out_gate() is True
    assert evidence["rank"] == 3
    assert evidence["output_gate_fused"] is False
    assert evidence["report"] == _passing_report()


def test_experiment_fails_closed_when_required_kernel_does_not_install(monkeypatch, tmp_path, chalk_experiment_module):
    report = _passing_report()
    report["fused_lora_delta"] = False
    package = _fake_chalk(monkeypatch, tmp_path, lambda module, **kwargs: report)
    monkeypatch.setattr(
        chalk_experiment_module,
        "CHALK_SOURCE_SHA256",
        chalk_experiment_module._python_source_digest(package),
    )
    monkeypatch.setenv(chalk_experiment_module.CHALK_EXPERIMENT_ENV, "1")

    with pytest.raises(RuntimeError, match="fused_lora_delta"):
        chalk_experiment_module.apply_chalk_experiment(_model(), rank=0)


def test_experiment_rejects_forbidden_report_entries(monkeypatch, tmp_path, chalk_experiment_module):
    report = _passing_report()
    report["fp8_dx"] = True
    package = _fake_chalk(monkeypatch, tmp_path, lambda module, **kwargs: report)
    monkeypatch.setattr(
        chalk_experiment_module,
        "CHALK_SOURCE_SHA256",
        chalk_experiment_module._python_source_digest(package),
    )
    monkeypatch.setenv(chalk_experiment_module.CHALK_EXPERIMENT_ENV, "1")

    with pytest.raises(RuntimeError, match="fp8_dx"):
        chalk_experiment_module.apply_chalk_experiment(_model(), rank=0)


def test_experiment_rejects_fused_output_gate(monkeypatch, tmp_path, chalk_experiment_module):
    package = _fake_chalk(monkeypatch, tmp_path, lambda module, **kwargs: _passing_report())
    attention = sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"].Qwen3_5Attention
    attention._chalk_out_gate_fused = True
    monkeypatch.setattr(
        chalk_experiment_module,
        "CHALK_SOURCE_SHA256",
        chalk_experiment_module._python_source_digest(package),
    )
    monkeypatch.setenv(chalk_experiment_module.CHALK_EXPERIMENT_ENV, "1")

    with pytest.raises(RuntimeError, match="output gate to remain eager"):
        chalk_experiment_module.apply_chalk_experiment(_model(), rank=0)


def test_experiment_rejects_wrong_source_and_model(monkeypatch, tmp_path, chalk_experiment_module):
    _fake_chalk(monkeypatch, tmp_path, lambda module, **kwargs: _passing_report())
    monkeypatch.setenv(chalk_experiment_module.CHALK_EXPERIMENT_ENV, "1")

    with pytest.raises(RuntimeError, match="dense qwen3_5 only"):
        chalk_experiment_module.apply_chalk_experiment(_model("qwen3_5_moe"), rank=0)

    with pytest.raises(RuntimeError, match="source digest mismatch"):
        chalk_experiment_module.apply_chalk_experiment(_model(), rank=0)


def test_experiment_rejects_ambiguous_flag(monkeypatch, chalk_experiment_module):
    monkeypatch.setenv(chalk_experiment_module.CHALK_EXPERIMENT_ENV, "true")

    with pytest.raises(ValueError, match="must be unset, 0, or 1"):
        chalk_experiment_module.apply_chalk_experiment(_model(), rank=0)


def test_transformer_build_applies_chalk_after_lora_and_before_fsdp():
    root = Path(__file__).parents[3]
    path = root / "verl/workers/engine/fsdp/transformer_impl.py"
    tree = ast.parse(path.read_text())
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FSDPEngine")
    build = next(
        node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "_build_model_optimizer"
    )
    calls = []
    for node in ast.walk(build):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        else:
            continue
        calls.append((node.lineno, name))
    lines = {name: line for line, name in calls}

    assert lines["_build_lora_module"] < lines["apply_chalk_experiment"] < lines["_build_fsdp_module"]
    assert lines["_apply_qat"] < lines["apply_chalk_experiment"]
