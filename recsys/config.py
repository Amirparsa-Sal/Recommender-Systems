"""Declarative hyper-parameter configs.

Every model declares its hyper-parameters as a dataclass built with
:func:`param`.  The very same declaration is used to

* create the command line flags automatically (``add_config_arguments``),
* validate parameters coming from a JSON experiment file
  (``config_from_dict``),
* document defaults and allowed values.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import typing
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def param(
    default: Any,
    help: str = "",
    choices: Optional[Sequence[Any]] = None,
    json: bool = False,
) -> Any:
    """Declare a hyper-parameter with a default value, help text and optional choices.

    ``json=True`` marks a structured value (a list or dict, annotated as ``list``):
    on the command line it is given as JSON text, ``@file.json`` or a comma separated
    list, in an experiment file it is written as a native JSON value.
    """
    metadata = {"help": help, "choices": tuple(choices) if choices else None, "json": json}
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda: copy.deepcopy(default), metadata=metadata)
    return field(default=default, metadata=metadata)


def parse_json_argument(text: str) -> Any:
    """Parse a structured command line value.

    Accepts JSON (``'[{"model": "item_knn"}]'``), ``@path/to/file.json`` or a plain
    comma separated list (``item_knn,rp3beta`` or ``1,0.5``) whose items become
    numbers where possible.
    """
    text = text.strip()
    if text.startswith("@"):
        text = Path(text[1:]).expanduser().read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    items = []
    for part in filter(None, (piece.strip() for piece in text.split(","))):
        try:
            items.append(float(part))
        except ValueError:
            items.append(part)
    return items


def _default_of(f: dataclasses.Field) -> Any:
    return f.default if f.default is not MISSING else f.default_factory()


@dataclass
class BaseConfig:
    """Parameters shared by every model."""

    seed: int = param(42, "Random seed used by the model.")

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check that every parameter restricted to a set of choices is valid."""
        for f in fields(self):
            choices = f.metadata.get("choices")
            value = getattr(self, f.name)
            if choices is not None and value not in choices:
                raise ValueError(
                    f"{type(self).__name__}.{f.name}={value!r} is invalid, "
                    f"choose one of {list(choices)}"
                )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def add_config_arguments(parser: Any, config_cls: type) -> None:
    """Add one ``--flag`` per dataclass field to an argparse parser or group.

    Underscores become dashes (``l1_ratio`` -> ``--l1-ratio``) and booleans get
    a ``--flag`` / ``--no-flag`` pair.
    """
    hints = typing.get_type_hints(config_cls)
    for f in fields(config_cls):
        flag = "--" + f.name.replace("_", "-")
        help_text = f"{f.metadata.get('help', '')} (default: %(default)s)"
        common = dict(dest=f.name, default=_default_of(f), help=help_text)
        hint = hints[f.name]
        if f.metadata.get("json"):
            parser.add_argument(flag, type=parse_json_argument, metavar="JSON", **common)
        elif hint is bool:
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, **common)
        else:
            parser.add_argument(
                flag, type=hint, choices=f.metadata.get("choices"), **common
            )


def config_from_namespace(config_cls: type, namespace: argparse.Namespace) -> Any:
    """Build a config instance from parsed command line arguments."""
    return config_cls(**{f.name: getattr(namespace, f.name) for f in fields(config_cls)})


def config_from_dict(config_cls: type, values: Mapping[str, Any]) -> Any:
    """Build a config instance from a plain dict, rejecting unknown keys."""
    hints = typing.get_type_hints(config_cls)
    structured = {f.name for f in fields(config_cls) if f.metadata.get("json")}
    known = {f.name for f in fields(config_cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {sorted(unknown)} for {config_cls.__name__}; "
            f"valid parameters: {sorted(known)}"
        )
    return config_cls(
        **{k: _coerce_structured(v, k) if k in structured else _coerce(hints[k], v, k)
           for k, v in values.items()}
    )


def _coerce_structured(value: Any, name: str) -> Any:
    """Structured parameters are lists / dicts; strings are parsed like command line values."""
    if isinstance(value, str):
        value = parse_json_argument(value)
    if not isinstance(value, (list, dict)):
        raise ValueError(f"Parameter '{name}' expects a JSON list or object, got {value!r}")
    return copy.deepcopy(value)


def _coerce(hint: type, value: Any, name: str) -> Any:
    """Convert JSON values to the declared type (e.g. ``10`` -> ``10.0``)."""
    if hint is bool:
        if isinstance(value, bool):
            return value
    elif hint is int:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif hint is str:
        if isinstance(value, str):
            return value
    raise ValueError(f"Parameter '{name}' expects {hint.__name__}, got {value!r}")
