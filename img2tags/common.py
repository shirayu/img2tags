#!/usr/bin/env python3
import collections.abc
import logging
from collections.abc import Iterable, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressType,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

CONSOLE: Console = Console(stderr=True)


def track(
    sequence: Sequence[ProgressType] | Iterable[ProgressType],
    *,
    prefix: str | None = None,
    total: int | None = None,
) -> Iterable[ProgressType]:
    cols = []
    if prefix is not None:
        cols.append(TextColumn(prefix))
    cols += [
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    progress = Progress(
        *cols,
        console=CONSOLE,
    )
    if total is None and isinstance(sequence, collections.abc.Sequence):
        total = len(sequence)
    with progress:
        yield from progress.track(
            sequence,
            description="",
            total=total,
        )


def setup_logging(log_level=logging.INFO):
    if logging.root.handlers:  # Already configured
        return
    handler = RichHandler(console=CONSOLE)

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)
