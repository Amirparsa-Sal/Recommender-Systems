"""Logging helpers."""

from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger (INFO by default, DEBUG with ``verbose``)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
