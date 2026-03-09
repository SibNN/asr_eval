from __future__ import annotations
from typing import Literal, Sequence
from dataclasses import dataclass, replace
import multiprocessing as mp
import copy

import numpy as np

from asr_eval.align.solvers.dynprog import solve_optimal_alignment
from asr_eval.utils.types import FLOATS
from asr_eval.streaming.model import (
    InputChunk,
    OutputChunk,
    Signal,
    StreamingASR,
    TranscriptionChunk,
    check_consistency,
)
from asr_eval.streaming.sender import (
    Cutoff, StreamingSender, get_uniform_cutoffs
)
from asr_eval.align.transcription import (
    Transcription, SingleVariantTranscription
)
from asr_eval.align.parsing import DEFAULT_PARSER
from asr_eval.align.matching import Match, MatchesList
from asr_eval.utils.misc import new_uid
from asr_eval.utils.audio_ops import resample


__all__ = [
    'StreamingEvaluationResults',
    'make_sender',
    'evaluate_streaming',
    'PartialAlignment',
    'StreamingASRErrorPosition',
    'get_audio_seconds_sent',
    'get_audio_seconds_processed',
    'get_partial_alignments',
    'remap_time',
]


@dataclass(kw_only=True)
class StreamingEvaluationResults:
    """A container for evaluation results for a streaming speech
    recognition on a single sample.
    
    Usually a result of the
    :func:`~asr_eval.streaming.evaluation.evaluate_streaming` function.
    """
    
    timed_transcription: Transcription
    """The ground truth transcription for the whole audio with filled
    timings for each token.
    """
    
    waveform: FLOATS
    """A waveform in float32 dtype with sampling rate 16000."""
    
    cutoffs: list[Cutoff]
    """A schedule on which the input chunks was sent."""
    
    input_chunks: list[InputChunk]
    """The input chunks history. The fields :code:`.put_timestamp` and
    :code:`.get_timestamp` are relative to the start time.
    """
    
    output_chunks: list[OutputChunk]
    """The output chunks history. The fields :code:`.put_timestamp` and
    :code:`.get_timestamp` are relative to the start time.
    """
    
    partial_alignments: list[PartialAlignment]
    """Alignments of the partial transcriptions against starting parts
    of the ground truth. Each partial alignment keep the
    :attr:`~asr_eval.streaming.evaluation.PartialAlignment.at_time`
    field that indicates a timestamp relative to the start time.
    """
    
    @property
    def start_timestamp(self) -> float:
        """A start time, where the first input chunk was put into the
        input buffer. Is always zero after time-normalisation performed
        by :func:`~asr_eval.streaming.evaluation.evaluate_streaming`.
        """
        
        return self.input_chunks[0].put_timestamp
    
    @property
    def finish_timestamp(self) -> float:
        """A finish time, where the last output chunk was put into the
        output buffer. The timestamp is relative to the starting
        moment.
        """
        
        return self.output_chunks[-1].put_timestamp


def make_sender(
    waveform: FLOATS,
    asr: StreamingASR,
    real_time_interval_sec: float = 1 / 5,
    speed_multiplier: float = 1,
    uid: str | None = None,
) -> tuple[list[Cutoff], StreamingSender]:
    """An automation to make a sender that sends an audio recording into
    a :class:`~asr_eval.streaming.model.StreamingASR`.

    After running :code:`cutoffs, sender = make_sender(...)`, you
    typically need to run :code:`sender.start_sending()` to start a
    thread that actually sends all the chunks.
    
    Args:
        waveform: The audio in float32 dtype with sampling rate 16000.
            Note that the streaming recognizer may accept a different
            sampling rate or dtype. A conversion to the required rate
            and dtype will be done on-the-fly inside this function.
        asr: A streaming transcriber to send chunks into.
        real_time_interval_sec: How often in real time to send chunks?
        speed_multiplier: For example, if :code:`speed_multiplier=2`,
            will send the audio twice of normal speed, that is, a 10
            seconds audio will be sent in 5 seconds.
        uid: Assign UID to the recording (select random of omitted).
    
    Returns:
        - A sending schedule in form of a list of cutoffs. See
            :func:`~asr_eval.streaming.sender.get_uniform_cutoffs` for
            details.
        - A sender object that is ready to start sending. Call
        :meth:`~asr_eval.streaming.sender.StreamingSender.start_sending`
        to start sending chunks.
    """
    assert asr.is_thread_started()
    uid = uid or new_uid()
    cutoffs = get_uniform_cutoffs(
        # this is crucial to do before resampling
        waveform=waveform,
        real_time_interval_sec=real_time_interval_sec,
        speed_multiplier=speed_multiplier,
    )
    waveform = resample(
        waveform,
        from_sampling_rate=16_000,
        to_sampling_rate=asr.sampling_rate,
    )
    cutoffs_resampled = [
        replace(c, arr_pos=int(c.arr_pos * asr.sampling_rate / 16_000))
        for c in cutoffs
    ]
    sender = StreamingSender(
        id=uid,
        verbose=False,
        cutoffs=cutoffs_resampled,
        waveform=waveform,
        asr=asr,
        sampling_rate=asr.sampling_rate,
    )
    return cutoffs, sender


def evaluate_streaming(
    timed_transcription: Transcription,
    waveform: FLOATS,
    cutoffs: list[Cutoff],
    input_chunks: list[InputChunk],
    output_chunks: list[OutputChunk],
    partial_alignment_interval: float = 0.25,
) -> StreamingEvaluationResults:
    """An automation to evaluate streaming recognition results.
    
    Aligns partial transcriptions against starting parts of the ground
    truth.
    
    For each of the :code:`timestamps` obtains the starting part of the
    :code:`timed_transcription` up to the specified timestamp, and
    aligns against the partial transcription that was received up to
    the specified timestamp. If the timestamp is inside a word in the
    ground truth transcription, considers two partial true
    transcriptions - with and without this word - and selects one with
    the best alignment score.
    
    Args:
        timed_transcription: The ground truth transcription for the
            whole audio with filled timings for each token. Is typically
            obtained with
            :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
        waveform: A waveform in float32 dtype with sampling rate 16000.
            Note that the streaming recognizer may accept a different
            sampling rate or dtype. A conversion to the required rate
            and dtype is typically done on-the-fly inside
            :func:`~asr_eval.streaming.evaluation.make_sender`
            function.
        cutoffs: A schedule on which the input chunks was sent.
        input_chunks: The input chunks history. Will create a copy of
            each chunk with relative timestamps instead of absolute.
            Will not modify the original chunks.
        output_chunks: The outputs chunks history. Will create a copy of
            each chunk with relative timestamps instead of absolute.
            Will not modify the original chunks.
        partial_alignment_interval: Time interval between consecutive
            alignments of the partial transcriptions against starting
            parts of the ground truth. Is real-timescale: for example,
            if a 10 sec long audios is transcribed for 30 seconds, and
            :code:`partial_alignment_interval=1`, then we will get
            30 partial alignments.
    
    Returns:
        A :class:`~asr_eval.streaming.evaluation.StreamingEvaluationResults`
        dataclass that scores the resulting partial alignments, as well
        as the input data.
    """
    # resetting time
    input_chunks = [copy.copy(c) for c in input_chunks]
    output_chunks = [copy.copy(c) for c in output_chunks]
    start_time = input_chunks[0].put_timestamp
    for chunk in input_chunks + output_chunks:
        chunk.put_timestamp -= start_time
        chunk.get_timestamp -= start_time

    # processing to save the results
    partial_alignments = get_partial_alignments(
        input_history=input_chunks,
        output_history=output_chunks,
        timed_transcription=timed_transcription,
        processes=1,
        timestamps=np.arange(
            input_chunks[0].put_timestamp,
            (
                output_chunks[-1].put_timestamp
                + partial_alignment_interval
                - 0.00001
            ),
            step=partial_alignment_interval,
        ).tolist(),
    )

    # cleaning large arrays to save the results
    for input_chunk in input_chunks:
        if input_chunk.data is not Signal.FINISH:
            input_chunk.data = None # type: ignore
        
    return StreamingEvaluationResults(
        timed_transcription=timed_transcription,
        waveform=waveform,
        partial_alignments=partial_alignments,
        cutoffs=cutoffs,
        input_chunks=input_chunks,
        output_chunks=output_chunks,
    )


@dataclass
class PartialAlignment:
    """
    An alignment between the ground truth up to the
    :code:`audio_seconds_sent` and the partial transcription.
    """
    
    pred: SingleVariantTranscription
    """A partial transcription from the streaming model. While the raw
    transcription is provided in form of the transcription chunks, this
    field represents the chunks joined with
    :meth:`~asr_eval.streaming.model.TranscriptionChunk.join` to form
    a transcription as text, and then parsed into words.
    """
    
    alignment: MatchesList
    """An alignment between the ground truth starting part, and the
    partial transcription.
    """
    
    at_time: float
    """The timestamp where the alignment was evaluated. All the output
    chunks sent later than this timestamp are not included.
    """
    
    audio_seconds_sent: float
    """How many seconds of the audio was sent by the time
    :attr:`~asr_eval.streaming.evaluation.PartialAlignment.at_time`.
    """
    
    audio_seconds_processed: float
    """How many seconds of the audio was processed by the time
    :attr:`~asr_eval.streaming.evaluation.PartialAlignment.at_time`.
    This value is extracted from the output chunks (see
    :attr:`~asr_eval.streaming.model.OutputChunk.seconds_processed`).
    """

    def get_error_positions(self) -> list[StreamingASRErrorPosition]:
        """Categorizes each word match from
        :attr:`~asr_eval.streaming.evaluation.PartialAlignment.alignment`
        into one of 5 types: ("correct", "deletion", "insertion",
        "replacement", "not_yet"). See the
        :class:`~asr_eval.streaming.evaluation.StreamingASRErrorPosition`
        docs for details.
        """
        
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
                assert match.true is not None
                results.append(StreamingASRErrorPosition(
                    start_time=match.true.start_time,
                    end_time=match.true.end_time,
                    sent_time=self.audio_seconds_sent,
                    processed_time=self.audio_seconds_processed,
                    status=match.status,
                ))
            elif match.status == 'insertion':
                left_pos = max(
                    [0] + [
                        match2.true.end_time
                        for match2 in head[:i]
                        if match2.true is not None
                    ]
                )
                right_pos = min(
                    [self.audio_seconds_processed]
                    + [
                        match2.true.start_time
                        for match2 in head[i + 1:]
                        if match2.true is not None
                    ]
                )
                results.append(StreamingASRErrorPosition(
                    start_time=left_pos,
                    end_time=right_pos,
                    sent_time=self.audio_seconds_sent,
                    processed_time=self.audio_seconds_processed,
                    status=match.status,
                ))
            else:
                assert match.true is not None
                results.append(StreamingASRErrorPosition(
                    start_time=match.true.start_time,
                    end_time=match.true.end_time,
                    sent_time=self.audio_seconds_sent,
                    processed_time=self.audio_seconds_processed,
                    status=match.status,
                ))
        
        # process tail
        for match in tail:
            assert match.true is not None
            results.append(StreamingASRErrorPosition(
                start_time=match.true.start_time,
                end_time=match.true.end_time,
                sent_time=self.audio_seconds_sent,
                processed_time=self.audio_seconds_processed,
                status='not_yet',
            ))
        
        return results


@dataclass
class StreamingASRErrorPosition:
    """A word-level match in a
    :class:`~asr_eval.streaming.evaluation.PartialAlignment` with
    assigned
    :attr:`~asr_eval.streaming.evaluation.StreamingASRErrorPosition.status`.
    """
    
    start_time: float
    """Start time of the ground truth word. If the match is insertion,
    no ground truth word exists, and the start time is the end time of
    the previous ground truth word, or zero.
    """
    
    end_time: float
    """End time of the ground truth word. If the match is insertion,
    no ground truth word exists, and the end time is the start time of
    the next ground truth word, or the processed time.
    """
    
    sent_time: float
    """How much seconds of the input audio was sent at the time when
    the current partial alignment was calculated.
    """
    
    processed_time: float
    """How much seconds of the input audio was processed at the time
    when the current partial alignment was calculated (see
    :attr:`~asr_eval.streaming.model.OutputChunk.seconds_processed`).
    """
    
    status: (
        Literal['correct', 'deletion', 'insertion', 'replacement', 'not_yet']
    )
    """One of 5 statuses: ("correct", "deletion", "insertion",
    "replacement", "not_yet"). The first 4 statuses are explained
    in the :attr:`~asr_eval.align.matching.Match.status`. The status
    "not_yet" is a special status that is assigned for trailing
    deletions. We consider that if a deletion is trailing, it represents
    a word not transcribed yet. This may occur either due to long
    inference times which cause delays, or because a model refuses to
    transcribe until it accumulates enough context. The field
    :attr:`~asr_eval.streaming.evaluation.StreamingASRErrorPosition.processed_time`
    allows to differentiate between these two reasons.
    """
    
    @property
    def center_time(self) -> float:
        """A center between the start and the end time."""
        return (self.start_time + self.end_time) / 2
    
    
def get_audio_seconds_sent(
    time: float, input_chunks: Sequence[InputChunk]
) -> float:
    """Given a full history of input chunks, and a :code:`time`, finds
    the last sent chunk with put timestamp before the :code:`time` and
    returns its :code:`.end_time`. If no such chunks, returns 0.
    """
    
    input_chunks_sent = [
        input_chunk for input_chunk in input_chunks
        if input_chunk.put_timestamp < time
    ]
    return input_chunks_sent[-1].end_time if input_chunks_sent else 0
    
    
def get_audio_seconds_processed(
    time: float, output_chunks: Sequence[OutputChunk]
) -> float:
    """Given a full history of output chunks, and a :code:`time`, finds
    the last sent chunk with put timestamp before :code:`time` and
    returns its
    :attr:`~asr_eval.streaming.model.OutputChunk.seconds_processed`. If
    no such chunks, returns 0.
    """
    
    output_chunks_sent = [
        output_chunk for output_chunk in output_chunks
        if output_chunk.put_timestamp < time
    ]
    return (
        output_chunks_sent[-1].seconds_processed if output_chunks_sent else 0
    )
    
    
def get_partial_alignments(
    input_history: Sequence[InputChunk],
    output_history: Sequence[OutputChunk],
    timed_transcription: Transcription,
    timestamps: list[float] | FLOATS | None = None,
    processes: int = 1,
) -> list[PartialAlignment]:
    """Aligns partial transcriptions against starting parts of the
    ground truth.
    
    For each of the :code:`timestamps` obtains ths starting part of the
    :code:`timed_transcription` up to the specified timestamp, and
    aligns against the partial transcription that was received up to
    the specified timestamp. If the timestamp is inside a word in the
    ground truth transcription, considers two partial true
    transcriptions - with and without this word - and selects one with
    the best alignment score.
    
    Args:
        input_history: The input chunks history.
        output_history: The output chunks history.
        timed_transcription: The ground truth transcription for the
            whole audio with filled timings for each token. Is typically
            obtained with
            :func:`~asr_eval.align.timings.fill_word_timings_inplace`.
        timestamps: A list of times when to evaluate partial results. If
            None, will evaluate after each of the output chunks, except
            the last :code:`Signal.FINISH` chunk if present.
        processes: If > 1, paralellizes using multiprocessing (we cannot
            use multithreading here because of GIL, considering that the
            alignment function is written on pure Python).
    """
    
    if output_history[-1].data is Signal.FINISH:
        output_history = output_history[:-1]

    # check that timings are not None and do not decrease
    assert np.all(np.diff([x.put_timestamp for x in input_history]) >= 0)
    assert np.all(np.diff([x.end_time for x in input_history]) >= 0)
    assert np.all(np.diff([x.put_timestamp for x in output_history]) >= 0)
    
    partial_alignments: list[PartialAlignment] = []
    for i, output_chunk in enumerate(output_history):
        partial_alignments.append(PartialAlignment(
            pred=DEFAULT_PARSER.parse_single_variant_transcription(
                TranscriptionChunk.join(output_history[:i + 1])
                # TODO optimize, this is a nested O(n^2) loop
                # however, typically should be fine
            ),
            alignment=None, # type: ignore
            at_time=output_chunk.put_timestamp,
            audio_seconds_sent=get_audio_seconds_sent(
                output_chunk.put_timestamp, input_history
            ),
            audio_seconds_processed=output_chunk.seconds_processed,
        ))
    
    if timestamps is not None:
        partial_alignments_for_times: list[PartialAlignment] = []
        for at_time in timestamps:
            prev_alignments = [
                pa for pa in partial_alignments if pa.at_time < at_time
            ]
            if len(prev_alignments):
                pa = copy.deepcopy(prev_alignments[-1])
                pa.at_time = at_time
                pa.audio_seconds_sent = (
                    get_audio_seconds_sent(at_time, input_history)
                )
            else:
                pa = PartialAlignment(
                    pred=SingleVariantTranscription('', tuple()),
                    alignment=None, # type: ignore
                    at_time=at_time,
                    audio_seconds_sent=get_audio_seconds_sent(
                        at_time, input_history
                    ),
                    audio_seconds_processed=0,
                )
            partial_alignments_for_times.append(pa)
        partial_alignments = partial_alignments_for_times
    
    def _partial_align(pa: PartialAlignment) -> MatchesList:
        true = timed_transcription.get_starting_part(pa.audio_seconds_sent)
        return solve_optimal_alignment(true, pa.pred)[0]

    if processes > 1:
        with mp.Pool(processes=processes) as pool:
            alignments = pool.map(_partial_align, partial_alignments)
    else:
        alignments = [_partial_align(pa) for pa in partial_alignments]
    
    for al, pa in zip(alignments, partial_alignments):
        pa.alignment = al
        
    return partial_alignments
        
        
        
def remap_time(
    cutoffs: list[Cutoff],
    input_chunks: list[InputChunk],
    output_chunks: list[OutputChunk]
) -> tuple[list[InputChunk], list[OutputChunk]]:
    """Remapping is an optional mechanism that eliminates time spans
    where both the sender waits (due to its schedule) and the model
    waits (because it already processed the chunk and waits for the
    next). This makes evaluation faster than real time with the same
    results. Using remapping is meaningful when input chunks was sent
    with :code:`without_delays=True`.
    
    Technically, :code:`remap_time` adds artificial delays in some
    places, shifting put timestamps and get timestamps forward for both
    input and output chunks. More concretely, it iterates chunks from
    the first to the last and finds input chunks that were
    taken from the input buffer until they should be placed in the
    buffer according to the :code:`cutoffs` schedule. When such a
    situation is found, all the put and get timestamps starting from
    this time are shifted forwards by the calculated time delta.
    
    In the end, this allows to imitate a chunk history as it would have
    looked if :code:`without_delays=False` in senders.
    
    Note:
        This erases any delay between :code:`get_timestamp` and
        :code:`put_timestamp` in output chunks.
    
    Note:
        This is not applicable (would work incorrectly) for
        :class:`~asr_eval.streaming.model.StreamingASR` that start
        another threads from its main background thread (where
        :attr:`~asr_eval.streaming.model.StreamingASR.is_multithreaded`
        is True).
    """
    
    check_consistency(input_chunks, output_chunks)
    
    input_chunks = copy.deepcopy(input_chunks)
    output_chunks = copy.deepcopy(output_chunks)

    inserted_delays: list[tuple[float, float]] = []

    # set put_timestamp as if StreamingAudioSender sends with correct delays
    start_time = input_chunks[0].put_timestamp
    for start_cutoff, input_chunk in (
        zip(cutoffs[:-1], input_chunks, strict=True)
    ):
        input_chunk.put_timestamp = start_time + start_cutoff.t_real

    # insert delays when get_timestamp < put_timestamp,
    # updating get_timestamp accordingly
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