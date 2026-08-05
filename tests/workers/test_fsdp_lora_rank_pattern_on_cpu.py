import ast
from pathlib import Path


def _attribute_path(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def test_fresh_lora_passes_rank_pattern_to_peft():
    source_path = Path(__file__).resolve().parents[2] / "verl/workers/engine/fsdp/transformer_impl.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        entries = {
            key.value: value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if {"target_parameters", "rank_pattern"} - entries.keys():
            continue
        rank_value = entries["rank_pattern"]
        assert isinstance(rank_value, ast.Call)
        assert _attribute_path(rank_value.func) == "convert_to_regular_types"
        assert len(rank_value.args) == 1
        assert _attribute_path(rank_value.args[0]) == "self.model_config.rank_pattern"
        return

    raise AssertionError("fresh lora config does not pass rank_pattern alongside target_parameters")
