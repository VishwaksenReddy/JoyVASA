# coding: utf-8

"""Helpers for merging CLI args into config dataclasses."""

from dataclasses import fields
from typing import Sequence, Type, TypeVar


ConfigT = TypeVar("ConfigT")


def _option_variants(field_name: str) -> set[str]:
    dashed = field_name.replace("_", "-")
    underscored = field_name
    return {
        f"--{dashed}",
        f"--{underscored}",
        f"--no-{dashed}",
        f"--no_{underscored}",
    }


def cli_option_was_provided(field_name: str, argv: Sequence[str]) -> bool:
    for arg in argv:
        for variant in _option_variants(field_name):
            if arg == variant or arg.startswith(f"{variant}="):
                return True
    return False


def build_config_from_cli(target_class: Type[ConfigT], args, argv: Sequence[str]) -> ConfigT:
    config = target_class()
    arg_values = vars(args)

    for field in fields(target_class):
        if field.name in arg_values and cli_option_was_provided(field.name, argv):
            setattr(config, field.name, arg_values[field.name])

    return config
