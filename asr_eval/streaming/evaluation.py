from __future__ import annotations

from typing import Literal, Sequence
from dataclasses import dataclass
import multiprocessing as mp
import copy

import numpy as np
import numpy.typing as npt


from asr_eval.streaming.buffer import ID_TYPE
from asr_eval.streaming.model import (
    InputChunk,
    OutputChunk,
    Signal,
    StreamingASR,
    TranscriptionChunk,
    check_consistency,
    prepare_audio_format,
)
from asr_eval.streaming.sender import BaseStreamingAudioSender, Cutoff, StreamingAudioSender
from asr_eval.streaming.caller import receive_full_transcription
from asr_eval.align.data import Match, MatchesList, MultiVariant, Token
from asr_eval.align.parsing import split_text_into_tokens
from asr_eval.align.partial import align_partial
from asr_eval.datasets.recording import Recording
from asr_eval.utils.misc import new_uid


@dataclass(kw_only=True)
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
    
    Properties:
    `start_timestamp` - a real time when the transcribing was started (the fisrst input chunk was sent).
    `finish_timestamp` - a real time when the transcribing was ended (the last output chunk was sent).
    """
    recording: Recording
    
    id: ID_TYPE
    sender: BaseStreamingAudioSender
    cutoffs: list[Cutoff]
    
    input_chunks: list[InputChunk]
    output_chunks: list[OutputChunk]
    
    partial_alignments: list[PartialAlignment]
    
    @property
    def start_timestamp(self) -> float:
        return self.input_chunks[0].put_timestamp
    
    @property
    def finish_timestamp(self) -> float:
        return self.output_chunks[-1].put_timestamp


def default_evaluation_pipeline(
    recording: Recording,
    asr: StreamingASR,
    real_time_interval_sec: float = 1 / 5,
    speed_multiplier: float = 1,
    without_delays: Literal['yes', 'yes_with_remapping', 'no'] = 'yes_with_remapping',
    partial_alignment_interval: float = 0.25,
    reset_timestamps: bool = True,
) -> RecordingStreamingEvaluation:
    '''
    A default pipeline to evaluate
    '''
    assert recording.waveform is not None
    
    # id = recording.hf_uid
    id = new_uid()

    # preparing input audio
    audio, array_len_per_sec = prepare_audio_format(recording.waveform, asr)
    
    # predicting
    sender = StreamingAudioSender(
        id=id,
        audio=audio,
        array_len_per_sec=array_len_per_sec,
        real_time_interval_sec=real_time_interval_sec,
        speed_multiplier=speed_multiplier,
        verbose=False,
    )
    output_chunks = receive_full_transcription(
        asr=asr,
        sender=sender,
        id=id,
        send_all_without_delays=without_delays in ('yes', 'yes_with_remapping'),
    )
    input_chunks = sender.history
    cutoffs = sender.get_send_times()
    
    # remapping
    if without_delays == 'yes_with_remapping':
        input_chunks, output_chunks = remap_time(cutoffs, sender.history, output_chunks)
    
    # resetting time
    if reset_timestamps:
        start_time = input_chunks[0].put_timestamp
        for chunk in input_chunks + output_chunks:
            chunk.put_timestamp -= start_time
            chunk.get_timestamp -= start_time

    # processing to save the results
    partial_alignments = get_partial_alignments(
        input_chunks,
        output_chunks,
        recording.transcription_words,
        processes=1,
        timestamps=np.arange(
            input_chunks[0].put_timestamp,
            output_chunks[-1].put_timestamp + partial_alignment_interval - 0.00001,
            step=partial_alignment_interval,
        ).tolist(),
    )

    # cleaning large arrays to save the results
    sender.audio = ''
    sender.history = []
    for input_chunk in input_chunks:
        if input_chunk.data != Signal.FINISH:
            input_chunk.data = ''
        
    return RecordingStreamingEvaluation(
        recording=recording,
        id=id,
        sender=sender,
        partial_alignments=partial_alignments,
        cutoffs=cutoffs,
        input_chunks=input_chunks,
        output_chunks=output_chunks,
    )


@dataclass
class PartialAlignment:
    """
    An optimal alignment between:
    - A part of spoken text up to `audio_seconds_processed`
    - A provieded partial transcription from the model
    
    Obtaining PartialAlignment requires word timings for the true transcription.
    """
    pred: list[Token]
    alignment: MatchesList
    at_time: float
    audio_seconds_sent: float
    audio_seconds_processed: float

    def get_error_positions(self) -> list[StreamingASRErrorPosition]:
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
                    + [token.start_time for match2 in head[i + 1:] for token in match2.true]
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
    
    
def get_audio_seconds_sent(time: float, input_chunks: Sequence[InputChunk]) -> float:
    input_chunks_sent = [
        input_chunk for input_chunk in input_chunks
        if input_chunk.put_timestamp < time
    ]
    return input_chunks_sent[-1].end_time if input_chunks_sent else 0
    
    
def get_audio_seconds_processed(time: float, output_chunks: Sequence[OutputChunk]) -> float:
    output_chunks_sent = [
        output_chunk for output_chunk in output_chunks
        if output_chunk.put_timestamp < time
    ]
    return output_chunks_sent[-1].seconds_processed if output_chunks_sent else 0
    
    
def get_partial_alignments(
    input_history: Sequence[InputChunk],
    output_history: Sequence[OutputChunk],
    true_word_timings: list[Token | MultiVariant],
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
    assert np.all(np.diff([x.put_timestamp for x in input_history])[1:] >= 0)
    assert np.all(np.diff([x.end_time for x in input_history])[1:] >= 0)
    assert np.all(np.diff([x.put_timestamp for x in output_history])[1:] >= 0)
    
    partial_alignments: list[PartialAlignment] = []
    for i, output_chunk in enumerate(output_history):
        partial_alignments.append(PartialAlignment(
            pred=split_text_into_tokens(TranscriptionChunk.join(output_history[:i + 1])),
            alignment=None, # type: ignore
            at_time=output_chunk.put_timestamp,
            audio_seconds_sent=get_audio_seconds_sent(output_chunk.put_timestamp, input_history),
            audio_seconds_processed=output_chunk.seconds_processed,
        ))
    
    if timestamps is not None:
        partial_alignments_for_times: list[PartialAlignment] = []
        for at_time in timestamps:
            prev_alignments = [pa for pa in partial_alignments if pa.at_time < at_time]
            if len(prev_alignments):
                pa = copy.deepcopy(prev_alignments[-1])
                pa.at_time = at_time
                pa.audio_seconds_sent = get_audio_seconds_sent(at_time, input_history)
            else:
                pa = PartialAlignment(
                    pred=[],
                    alignment=None, # type: ignore
                    at_time=at_time,
                    audio_seconds_sent=get_audio_seconds_sent(at_time, input_history),
                    audio_seconds_processed=0,
                )
            partial_alignments_for_times.append(pa)
        partial_alignments = partial_alignments_for_times
    
    if processes > 1:
        pool = mp.Pool(processes=processes)
        alignments = pool.map(
            lambda pa: align_partial(true_word_timings, pa.pred, pa.audio_seconds_processed),
            partial_alignments
        )
    else:
        alignments = [
            align_partial(true_word_timings, pa.pred, pa.audio_seconds_processed)
            for pa in partial_alignments
        ]
    
    for al, pa in zip(alignments, partial_alignments):
        pa.alignment = al
        
    return partial_alignments
        
        
        
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
    start_time = input_chunks[0].put_timestamp
    for start_cutoff, input_chunk in zip(cutoffs[:-1], input_chunks, strict=True):
        input_chunk.put_timestamp = start_time + start_cutoff.t_real

    # insert delays when get_timestamp < put_timestamp, updating get_timestamp accordingly
    for input_chunk in input_chunks[1:]:
        put = input_chunk.put_timestamp
        get = input_chunk.get_timestamp
        get += sum([delta for t, delta in inserted_delays if t <= get])
        if get < put:
            delay = put - get
            inserted_delays.append((input_chunk.get_timestamp, delay))
            get += delay
        
        input_chunk.get_timestamp = get

    # update put_timestamp and get_timestamp for output chunks accordingly
    for output_chunk in output_chunks:
        put = output_chunk.put_timestamp
        put += sum([delta for t, delta in inserted_delays if t <= put])
        output_chunk.put_timestamp = put
        output_chunk.get_timestamp = put
        
    check_consistency(input_chunks, output_chunks)
    
    return input_chunks, output_chunks