import re
from typing import Literal, Sequence
from dataclasses import dataclass
from functools import partial
import multiprocessing as mp

from gigaam.model import GigaAMASR
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .model import InputChunk, OutputChunk, TranscriptionChunk
from ..ctc.base import ctc_mapping
from ..ctc.forced_alignment import forced_alignment
from ..models.gigaam import transcribe_with_gigaam_ctc, encode, decode, FREQ
from ..align.data import MatchesList, Token
from ..align.parsing import split_text_into_tokens
from ..align.recursive import align


@dataclass
class PartialAlignment:
    alignment: MatchesList
    audio_seconds_sent: float
    audio_seconds_processed: float | None = None
    real_seconds_overhead: float = 0
    
    
def _calculate_partial_alignment(
    input_history: Sequence[InputChunk],
    output_history: Sequence[OutputChunk],
    true_word_timings: list[Token],
    output_chunk_idx: int,
) -> PartialAlignment:
    output_chunk = output_history[output_chunk_idx]
    text = TranscriptionChunk.join(output_history[:output_chunk_idx + 1])

    pred_tokens = split_text_into_tokens(text)

    chunks_sent = [
        input_chunk for input_chunk in input_history
        if input_chunk.put_timestamp < output_chunk.put_timestamp # pyright: ignore[reportOperatorIssue]
    ]
    audio_seconds_sent: float = chunks_sent[-1].end_time if chunks_sent else 0 # pyright: ignore[reportAssignmentType]

    n_true_words, in_true_word = words_count(
        true_word_timings,
        audio_seconds_sent if output_chunk.seconds_processed is None else output_chunk.seconds_processed
    )

    option1 = true_word_timings[:n_true_words]
    alignment = align(option1, pred_tokens) # pyright: ignore[reportArgumentType]

    if in_true_word:
        option2 = true_word_timings[:n_true_words + 1]
        alignment2 = align(option2, pred_tokens) # pyright: ignore[reportArgumentType]
        
        if alignment2.score > alignment.score:
            alignment = alignment2

    return PartialAlignment(
        alignment=alignment,
        audio_seconds_sent=audio_seconds_sent,
        audio_seconds_processed=output_chunk.seconds_processed,
        real_seconds_overhead=(
            output_chunk.put_timestamp - chunks_sent[-1].put_timestamp # type: ignore
            if chunks_sent and len(chunks_sent) == len(input_history)
            else 0
        ),
    )


def get_partial_alignments(
    input_history: Sequence[InputChunk],
    output_history: Sequence[OutputChunk],
    true_word_timings: list[Token],
    processes: int = 1,
) -> list[PartialAlignment]:
    # check that timings are not None and do not decrease
    assert np.all(np.diff([(x.put_timestamp or np.nan) for x in input_history])[1:] >= 0)
    assert np.all(np.diff([(x.start_time or np.nan) for x in input_history])[1:] >= 0)
    assert np.all(np.diff([(x.end_time or np.nan) for x in input_history])[1:] >= 0)
    assert np.all(np.diff([(x.put_timestamp or np.nan) for x in output_history])[1:] >= 0)
    
    if processes > 1:
        pool = mp.Pool(processes=processes)
        return pool.map(
            partial(_calculate_partial_alignment, input_history, output_history, true_word_timings),
            range(len(output_history))
        )
    else:
        return [
            _calculate_partial_alignment(input_history, output_history, true_word_timings, i)
            for i in range(len(output_history))
        ]
        

def partial_alignment_diagram(
    partial_alignments: list[PartialAlignment],
    true_word_timings: list[Token],
    audio_len: float,
    figsize: tuple[float, float] = (15, 15),
    y_type: Literal['sent', 'processed'] = 'sent',
):
    plt.figure(figsize=figsize) # type: ignore

    # main lines
    plt.plot([0, audio_len], [0, audio_len], color='lightgray') # type: ignore
    plt.plot([0, audio_len], [0, 0], color='lightgray') # type: ignore

    # word timings
    for token in true_word_timings:
        plt.fill_between( # type: ignore
            [token.start_time, token.end_time],
            [0, 0],
            [audio_len, audio_len],
            color='#eeeeee',
            zorder=-1
        )
        plt.text( # type: ignore
            (token.start_time + token.end_time) / 2,
            0,
            ' ' + str(token.value),
            fontsize=10,
            rotation=90,
            ha='center',
            va='bottom',
        )

    # partial alignments
    last_end_time = 0
    for partial_alignment in partial_alignments:
        y_pos = (
            partial_alignment.audio_seconds_sent
            if y_type == 'sent'
            else partial_alignment.audio_seconds_processed
        )
        for match in partial_alignment.alignment.matches:
            if len(match.true) == 0:
                plt.scatter([last_end_time], [y_pos], color='black', s=10, zorder=2) # type: ignore
            else:
                assert len(match.true) == 1
                last_end_time = match.true[0].end_time

                status = match.get_status()
                skip = False

                if status == 'correct':
                    color = 'green'
                elif status == 'replacement':
                    color = 'red'
                else:
                    assert status == 'deletion'
                    color = None
                    skip = True
                
                if not skip:
                    plt.plot([start, end], [y_pos, y_pos], color=color) # type: ignore
        if partial_alignment.audio_seconds_processed is not None:
            plt.scatter( # type: ignore
                [partial_alignment.audio_seconds_processed], [y_pos], # type: ignore
                s=20, zorder=2, color='gray', marker='|'
            )

    plt.show() # type: ignore


def get_word_timings(
    model: GigaAMASR,
    waveform: npt.NDArray[np.floating],
    text: str | None = None,
) -> list[Token]:
    '''
    Outputs a list of words and their timings in seconds:

    ([('и', 0.12, 0.16),
        ('поэтому', 0.2, 0.56),
        ('использовать', 0.64, 1.28),
        ('их', 1.32, 1.44),
        ('в', 1.48, 1.56),
        ('повседневности', 1.6, 2.36),
    '''
    outputs = transcribe_with_gigaam_ctc(model, [waveform])[0]
    if text is None:
        tokens = outputs.log_probs.argmax(axis=1)
    else:
        tokens, _probs = forced_alignment(
            outputs.log_probs,
            encode(model, text),
            blank_id=model.decoding.blank_id
        )
    letter_per_frame = decode(model, tokens)
    word_timings = [
        Token(
            value=''.join(ctc_mapping(list(match.group()), blank='_')),
            start_time=match.start() / FREQ,
            end_time=match.end() / FREQ,
        )
        for match in re.finditer(r'[а-я]([а-я_]*[а-я])?', letter_per_frame)
    ]

    # fill positions
    if text is not None:
        true_tokens_with_positions = split_text_into_tokens(text)
        for token_to_return, token_with_pos in zip(
            word_timings, true_tokens_with_positions, strict=True
        ):
            assert token_to_return.value == token_with_pos.value
            token_to_return.pos = token_with_pos.pos

    return word_timings


def words_count(
    word_timings: Sequence[Token],
    time: float,
) -> tuple[int, bool]:
    '''
    Returns a tuple of:
    1. Number of full words in the time span [0, time]
    2. `in_word` flag: is the given time inside a word?
    '''
    count = 0
    in_word = False

    for token in word_timings:
        if token.end_time <= time:
            count += 1
        else:
            if token.start_time < time:
                in_word = True
            break
    
    return count, in_word