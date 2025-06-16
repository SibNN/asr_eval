import re
from typing import Sequence, cast
from dataclasses import dataclass
from functools import partial
import multiprocessing as mp
import copy

from gigaam.model import GigaAMASR
import numpy as np
import numpy.typing as npt

from .model import InputChunk, OutputChunk, TranscriptionChunk, check_consistency
from .sender import Cutoff
from ..ctc.base import ctc_mapping
from ..ctc.forced_alignment import forced_alignment
from ..models.gigaam import transcribe_with_gigaam_ctc, encode, decode, FREQ
from ..align.data import MatchesList, Token
from ..align.parsing import split_text_into_tokens
from ..align.recursive import align


@dataclass
class PartialAlignment:
    """
    An optimal alignment between:
    - A part of spoken text up to `audio_seconds_processed`
    - A provieded partial transcription from the model
    
    Obtaining PartialAlignment requires word timings for the true transcription.
    """
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
    """A helper function to paralellize get_partial_alignments() using multiprocessing."""
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
    """
    Accepts:
    - A full history of input and output chunks from StreamASR (belonging to the same recording ID)
    - A true transcription as a list of Token with `.start_time` and `.end_time` filled
    
    For each output chunk calculates a PartialAlignment. To do this, obtains a partial true
    transcription based on `output_chunk.seconds_processed` and aligns with the predicted transcription
    (all the previous output chunks joined). If the time is inside a word, that is `token.start_time
    < output_chunk.seconds_processed < token.end_time`, considers two partial true transcription - with
    and without this word - and selects one with the best alignment score.
    
    Can be paralellized using multiprocessing if `processes > 1` (we cannot use multithreading here
    because of GIL, considering that the alignment function is written on pure Python).
    """
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
        

def get_word_timings(
    model: GigaAMASR,
    waveform: npt.NDArray[np.floating],
    text: str | None = None,
    normalize: bool = True,
) -> list[Token]:
    '''
    
    Using GigaAM CTC model, performs either a forced alignment or an argmax prediction, and returns
    a list of words and their timings in seconds:

    [
        (Token('и'), 0.12, 0.16),
        (Token('поэтому'), 0.2, 0.56),
        (Token('использовать'), 0.64, 1.28),
        (Token('их'), 1.32, 1.44),
        (Token('в'), 1.48, 1.56),
        (Token('повседневности'), 1.6, 2.36),
        ...
    ]
    '''
    outputs = transcribe_with_gigaam_ctc(model, [waveform])[0]
    if text is None:
        tokens = outputs.log_probs.argmax(axis=1)
    else:
        if normalize:
            text = text.lower().replace('ё', 'е').replace('-', ' ')
            for char in ('.', ',', '!', '?', ';', ':', '"', '(', ')'):
                text = text.replace(char, '')
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
            token_to_return.start_pos = token_with_pos.start_pos
            token_to_return.end_pos = token_with_pos.end_pos

    return word_timings


def words_count(
    word_timings: Sequence[Token],
    time: float,
) -> tuple[int, bool]:
    '''
    Given a list of Token with `.start_time` and `.end_time` filled, returns a tuple of:
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


def remap_time(
    cutoffs: list[Cutoff],
    input_chunks: list[InputChunk],
    output_chunks: list[OutputChunk]
) -> tuple[list[InputChunk], list[OutputChunk]]:
    """
    Accept the result of a transcription of a single recoding, obtained using
    `BaseStreamingAudioSender.send_all_without_delays()`.
    
    Makes a deep copy of input and output chunks. Then modifies put and get timestamps
    in both input and output chunks  to simulate the result of `.start_sending()` instead of
    `.send_all_without_delays()`.
    """
    fl = partial(cast, float)
    
    check_consistency(input_chunks, output_chunks)
    
    input_chunks = copy.deepcopy(input_chunks)
    output_chunks = copy.deepcopy(output_chunks)

    inserted_delays: list[tuple[float, float]] = []

    # set put_timestamp as if StreamingAudioSender sends with correct delays
    start_time = cast(float, input_chunks[0].put_timestamp)
    for start_cutoff, input_chunk in zip(cutoffs[:-1], input_chunks, strict=True):
        input_chunk.put_timestamp = start_time + start_cutoff.t_real

    # insert delays when get_timestamp < put_timestamp, updating get_timestamp accordingly
    for input_chunk in input_chunks[1:]:
        put = fl(input_chunk.put_timestamp)
        get = fl(input_chunk.get_timestamp)
        get += sum([delta for t, delta in inserted_delays if t <= get])
        if get < put:
            delay = put - get
            inserted_delays.append((fl(input_chunk.get_timestamp), delay))
            get += delay
        
        input_chunk.get_timestamp = get

    # update put_timestamp and get_timestamp for output chunks accordingly
    for output_chunk in output_chunks:
        put = fl(output_chunk.put_timestamp)
        put += sum([delta for t, delta in inserted_delays if t <= put])
        output_chunk.put_timestamp = put
        output_chunk.get_timestamp = put
        
    check_consistency(input_chunks, output_chunks)
    
    return input_chunks, output_chunks