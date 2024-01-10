#!/usr/bin/env python3


import collections.abc
import sys
from typing import Iterable, Optional, Sequence, Union

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressType,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def track(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    *,
    prefix: Optional[str] = None,
    total: Optional[int] = None,
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
        console=Console(file=sys.stderr),
    )
    if total is None and isinstance(sequence, collections.abc.Sequence):
        total = len(sequence)
    with progress:
        yield from progress.track(
            sequence,
            description="",
            total=total,
        )
