from datetime import timedelta

import srt


def utterances_to_srt(utterances: list[tuple[str, float, float]]) -> str:
    '''
    Composes an SRT file.
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