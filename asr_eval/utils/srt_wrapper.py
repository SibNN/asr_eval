from datetime import timedelta
from pathlib import Path

import srt

from asr_eval.segments.segment import TimedText


__all__ = [
    'utterances_to_srt',
    'read_srt',
]


def utterances_to_srt(utterances: list[tuple[str, float, float]]) -> str:
    """Composes an SRT file contents from texts, start and end times."""
    
    return srt.compose([ # type: ignore
        srt.Subtitle(
            index=None,
            start=timedelta(seconds=start),
            end=timedelta(seconds=end),
            content=text
        )
        for text, start, end in utterances
    ], reindex=True)



def read_srt(path: str | Path) -> list[TimedText]:
    """Reads .srt transcription file into a list of :code:`TimedText`."""

    return [
        TimedText(
            x.start.total_seconds(),
            x.end.total_seconds(),
            x.content,
        )
        for x in srt.parse(Path(path).read_text()) # pyright: ignore[reportUnknownMemberType]
    ]