from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(
    base: dict, overrides: dict,
) -> dict:
    """Deep-merge *overrides* into *base* copy."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        both_dicts = (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        )
        if both_dicts:
            result[key] = merge_configs(
                result[key], value,
            )
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_cli_overrides(
    overrides: list[str] | None,
) -> dict:
    """Convert 'key.sub=val' strings to nested dict."""
    if not overrides:
        return {}

    result: dict = {}
    for item in overrides:
        dotted_key, raw = item.split("=", 1)
        parsed = _safe_parse_value(raw)

        keys = dotted_key.split(".")
        target = result
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = parsed

    return result


def _safe_parse_value(raw: str) -> Any:
    """Parse via ast.literal_eval, fallback to str."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw
