from __future__ import annotations

from typing import Literal, Sequence, cast
from dataclasses import dataclass
import multiprocessing as mp
import copy

import numpy as np
import numpy.typing as npt


from .buffer import ID_TYPE
from .model import InputChunk, OutputChunk, Signal, StreamingASR, TranscriptionChunk, check_consistency
from .sender import BaseStreamingAudioSender, Cutoff, StreamingAudioSender
from .caller import receive_full_transcription
from ..align.data import Match, MatchesList, Token
from ..align.parsing import split_text_into_tokens
from ..align.recursive import match_from_pair
from ..align.partial import align_partial, words_count
from ..utils import N
from ..data import Recording


@dataclass
class RecordingStreamingEvaluation:
    """
    Keeps data related to obtaining and evaluating streaming prediction for some model and
    `asr_eval.data.Recording`.
    
    `id` - a unique ID to send into StreamingASR (is required for StreamingASR to work).
    `sender` - a sender that will send input chunks and store them as `.history`
    `cutoffs` - a cutoffs obtained by `sender.get_send_times()`
    
    `input_chunks` - input chunks obtained from StreamingASR
    `output_chunks` - output chunks obtained from StreamingASR
    
    `partial_alignments` - partial alignments obtained from input and output chunks
    """
    id: ID_TYPE | None = None
    sender: BaseStreamingAudioSender | None = None
    cutoffs: list[Cutoff] | None = None
    
    input_chunks: list[InputChunk] | None = None
    output_chunks: list[OutputChunk] | None = None
    
    @property
    def start_timestamp(self) -> float:
        return N(N(self.input_chunks)[0].put_timestamp)
    
    @property
    def finish_timestamp(self) -> float:
        return N(N(self.output_chunks)[-1].put_timestamp)
    
    partial_alignments: list[PartialAlignment] | None = None


def default_evaluation_pipeline(
    recording: Recording,
    asr: StreamingASR,
) -> RecordingStreamingEvaluation:
    assert recording.waveform is not None

    evals = RecordingStreamingEvaluation()
    evals.id = recording.hf_uid

    # preparing input audio
    match asr.audio_type:
        case 'float':
            audio = recording.waveform
            array_len_per_sec = asr.sampling_rate
        case 'int':
            audio = (recording.waveform * 32768).astype(np.int16)
            array_len_per_sec = asr.sampling_rate
        case 'bytes':
            audio = (recording.waveform * 32768).astype(np.int16).tobytes()
            array_len_per_sec = asr.sampling_rate * 2  # x2 because of the conversion int16 -> bytes
    
    # predicting
    evals.sender = StreamingAudioSender(
        id=evals.id,
        audio=audio,
        array_len_per_sec=array_len_per_sec,
        real_time_interval_sec=1 / 5,
        speed_multiplier=1,
        verbose=False,
    )
    output_chunks = receive_full_transcription(
        asr=asr,
        sender=evals.sender,
        id=evals.id,
        send_all_without_delays=True,
    )

    # processing to save the results
    evals.cutoffs = evals.sender.get_send_times()
    evals.input_chunks, evals.output_chunks = remap_time(
        evals.cutoffs,
        evals.sender.history,
        output_chunks,
    )
    evals.partial_alignments = get_partial_alignments(
        evals.input_chunks,
        evals.output_chunks,
        cast(list[Token], recording.transcription_words),
        processes=1,
        timestamps=np.arange(
            evals.start_timestamp,
            evals.finish_timestamp + 0.2 - 0.0001,
            step=0.2,
        ).tolist(),
    )

    # cleaning large arrays to save the results
    evals.sender.audio = ''
    evals.sender.history = []
    for input_chunk in evals.input_chunks:
        if input_chunk.data != Signal.FINISH:
            input_chunk.data = ''
        
    return evals


@dataclass
class PartialAlignment:
    """
    An optimal alignment between:
    - A part of spoken text up to `audio_seconds_processed`
    - A provieded partial transcription from the model
    
    Obtaining PartialAlignment requires word timings for the true transcription.
    """
    alignment: MatchesList
    at_time: float
    audio_seconds_sent: float
    audio_seconds_processed: float | None = None

    def get_error_positions(self) -> list[StreamingASRErrorPosition]:
        assert self.audio_seconds_processed is not None

        results: list[StreamingASRErrorPosition] = []

        # split into head and tail
        head: list[Match] = []
        tail: list[Match] = []
        in_tail = True
        for match in self.alignment.matches[::-1]:
            in_tail &= match.status == 'deletion'
            if in_tail:
                tail.insert(0, match)
            else:
                head.insert(0, match)
        
        # process head
        for i, match in enumerate(head):
            if match.status == 'correct':
                for token in match.true:
                    results.append(StreamingASRErrorPosition(
                        start_time=token.start_time,
                        end_time=token.end_time,
                        processed_time=self.audio_seconds_processed,
                        status=match.status,
                    ))
            elif match.status == 'insertion':
                left_pos = max(
                    [0] + [token.end_time for match2 in head[:i] for token in match2.true]
                )
                right_pos = min(
                    [self.audio_seconds_processed]
                    + [token.end_time for match2 in head[i + 1:] for token in match2.true]
                )
                results.append(StreamingASRErrorPosition(
                    start_time=left_pos,
                    end_time=right_pos,
                    processed_time=self.audio_seconds_processed,
                    status=match.status,
                ))
            else:
                for token in match.true:
                    results.append(StreamingASRErrorPosition(
                        start_time=token.start_time,
                        end_time=token.end_time,
                        processed_time=self.audio_seconds_processed,
                        status=match.status,
                    ))
        
        # process tail
        for match in tail:
            for token in match.true:
                results.append(StreamingASRErrorPosition(
                    start_time=token.start_time,
                    end_time=token.end_time,
                    processed_time=self.audio_seconds_processed,
                    status='not_yet',
                ))
        
        return results


@dataclass
class StreamingASRErrorPosition:
    start_time: float
    end_time: float
    processed_time: float
    status: Literal['correct', 'deletion', 'insertion', 'replacement', 'not_yet']

    @property
    def time_delta(self) -> float:
        return self.processed_time -  (self.start_time + self.end_time) / 2
    
    
def get_partial_alignments(
    input_history: Sequence[InputChunk],
    output_history: Sequence[OutputChunk],
    true_word_timings: list[Token],
    timestamps: list[float] | npt.NDArray[np.integer] | None = None,
    processes: int = 1,
) -> list[PartialAlignment]:
    """
    Accepts:
    - A full history of input and output chunks from StreamASR (belonging to the same recording ID)
    - A true transcription as a list of Token with `.start_time` and `.end_time` filled
    
    For each output chunk except the last chunk with Signal.FINISH (if present) calculates a
    PartialAlignment. To do this, the method obtains a partial true
    transcription based on `output_chunk.seconds_processed` and aligns with the predicted transcription
    (all the previous output chunks joined). If the time is inside a word, that is `token.start_time
    < output_chunk.seconds_processed < token.end_time`, considers two partial true transcription - with
    and without this word - and selects one with the best alignment score.
    
    If `timestamps` is specified, calculates PartialAlignment for each time, not for each output chunk. For
    each time, joins all output chunks with `.put_timestamp` less than the specified time.
    
    Can be paralellized using multiprocessing if `processes > 1` (we cannot use multithreading here
    because of GIL, considering that the alignment function is written on pure Python).
    """
    if output_history[-1].data is Signal.FINISH:
        output_history = output_history[:-1]

    # check that timings are not None and do not decrease
    assert np.all(np.diff([(x.put_timestamp or np.nan) for x in input_history])[1:] >= 0)
    assert np.all(np.diff([(x.start_time or np.nan) for x in input_history])[1:] >= 0)
    assert np.all(np.diff([(x.end_time or np.nan) for x in input_history])[1:] >= 0)
    assert np.all(np.diff([(x.put_timestamp or np.nan) for x in output_history])[1:] >= 0)
    
    texts: list[list[Token]] = [
        split_text_into_tokens(TranscriptionChunk.join(output_history[:i + 1]))
        for i in range(len(output_history))
    ]
    seconds_processed: list[float] = [
        N(output_chunk.seconds_processed)
        for output_chunk in output_history
    ]
    
    if processes > 1:
        pool = mp.Pool(processes=processes)
        alignments = pool.map(
            lambda x: align_partial(true_word_timings, *x),
            zip(texts, seconds_processed, strict=True) 
        )
    else:
        alignments = [
            align_partial(true_word_timings, *x)
            for x in zip(texts, seconds_processed, strict=True) 
        ]
    
    partial_alignments: list[PartialAlignment] = []
    for al, output_chunk in zip(alignments, output_history):
        input_chunks_sent = [
            input_chunk for input_chunk in input_history
            if N(input_chunk.put_timestamp) < N(output_chunk.put_timestamp)
        ]
        partial_alignments.append(PartialAlignment(
            alignment=al,
            at_time=N(output_chunk.put_timestamp),
            audio_seconds_sent=N(input_chunks_sent[-1].end_time) if input_chunks_sent else 0,
            audio_seconds_processed=N(output_chunk.seconds_processed),
        ))
    
    if timestamps is None:
        return partial_alignments
    
    partial_alignments_for_times: list[PartialAlignment] = []
    for time in timestamps:
        output_chunks_put = [x for x in output_history if N(x.put_timestamp) < time]
        n_chunks_sent = len(output_chunks_put)
        if n_chunks_sent > 0:
            partial_alignment = copy.deepcopy(partial_alignments[n_chunks_sent - 1])
            partial_alignment.at_time = time
        else:
            partial_alignment = PartialAlignment(
                alignment=MatchesList.from_list([]),
                at_time=time,
                audio_seconds_sent=0,
                audio_seconds_processed=0,
            )
        input_chunks_sent = [x for x in input_history if N(x.put_timestamp) < time]
        partial_alignment.audio_seconds_sent = (
            max([N(x.end_time) for x in input_chunks_sent])
            if len(input_chunks_sent)
            else 0
        )
        n_sent_words, _ = words_count(true_word_timings, partial_alignment.audio_seconds_sent)
        sent_words = true_word_timings[:n_sent_words]
        n_processed_words = sum([len(m.true) for m in partial_alignment.alignment.matches])
        for word in sent_words[n_processed_words:]:
            partial_alignment.alignment = (
                partial_alignment.alignment.append(match_from_pair([word], []))
            )
        
        partial_alignments_for_times.append(partial_alignment)
    
    return partial_alignments_for_times
        
        
        
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
        put = N(input_chunk.put_timestamp)
        get = N(input_chunk.get_timestamp)
        get += sum([delta for t, delta in inserted_delays if t <= get])
        if get < put:
            delay = put - get
            inserted_delays.append((N(input_chunk.get_timestamp), delay))
            get += delay
        
        input_chunk.get_timestamp = get

    # update put_timestamp and get_timestamp for output chunks accordingly
    for output_chunk in output_chunks:
        put = N(output_chunk.put_timestamp)
        put += sum([delta for t, delta in inserted_delays if t <= put])
        output_chunk.put_timestamp = put
        output_chunk.get_timestamp = put
        
    check_consistency(input_chunks, output_chunks)
    
    return input_chunks, output_chunks