from datetime import timedelta

import srt


__all__ = [
    'utterances_to_srt',
]


def utterances_to_srt(utterances: list[tuple[str, float, float]]) -> str:
    '''
    Composes an SRT file contents from texts, start and end times.
    '''
    return srt.compose([ # type: ignore
        srt.Subtitle(
            index=None,
            start=timedelta(seconds=start),
            end=timedelta(seconds=end),
            content=text
        )
        for text, start, end in utterances
    ], reindex=True)