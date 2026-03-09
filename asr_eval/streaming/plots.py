from typing import Literal
from itertools import pairwise

import numpy as np
import matplotlib.pyplot as plt

from asr_eval.align.plots import draw_timed_transcription
from asr_eval.streaming.model import InputChunk, OutputChunk, Signal
from asr_eval.streaming.evaluation import (
    PartialAlignment,
    StreamingEvaluationResults,
    get_audio_seconds_processed,
    get_audio_seconds_sent,
)
from asr_eval.utils.types import INTS


__all__ = [
    'partial_alignments_plot',
    'visualize_history',
    'streaming_error_vs_latency_histogram',
    'latency_plot',
    'show_last_alignments',
]


def draw_partial_alignment(
    partial_alignment: PartialAlignment,
    override_y_pos: float | None = None,
    ax: plt.Axes | None = None,
    show_processed_time: bool = True,
    show_send_time: bool = True,
):
    """Draw a single partial alignment."""
    
    ax = ax or plt.gca()
    y_pos = (
        override_y_pos
        if override_y_pos is not None
        else partial_alignment.at_time
    )
    insertions_x: list[float] = []
    for pos in partial_alignment.get_error_positions():
        if pos.status == 'insertion':
            insertions_x.append((pos.start_time + pos.end_time) / 2)
        else:
            match pos.status:
                case 'correct':
                    color = 'green'
                case 'deletion':
                    color = 'red'
                case 'replacement':
                    color = 'red'
                case 'not_yet':
                    color = 'gray'
            dtime = min((pos.end_time - pos.start_time) / 8, 0.2)
            ax.plot( # type: ignore
                [pos.start_time + dtime, pos.end_time - dtime], [y_pos, y_pos],
                color=color
            )
    insertions_x = list(set(insertions_x))  # deduplicate, TODO print count
    ax.scatter( # type: ignore
        insertions_x, [y_pos] * len(insertions_x),
        color='darkred', s=10, zorder=3,
    )
    if show_processed_time:
        ax.scatter( # type: ignore
            [partial_alignment.audio_seconds_processed], [y_pos], # type: ignore
            s=30, zorder=2, color='red', marker='|'
        )
    if show_send_time:
        ax.scatter( # type: ignore
            [partial_alignment.audio_seconds_sent], [y_pos], # type: ignore
            s=20, zorder=2, color='red', marker='.'
        )


def partial_alignments_plot(
    eval: StreamingEvaluationResults,
    ax: plt.Axes | None = None,
):
    """Draw a partial alignment diagram.
    
    See more details and examples in the user guide:
    :doc:`/guide_streaming_evaluation`.
    """
    
    ax = ax or plt.gca()
    
    y1, y2 = eval.start_timestamp, eval.finish_timestamp
    y0 = y1 - (y2 - y1) / 15
    
    draw_timed_transcription(
        eval.timed_transcription,
        y_pos=y0,
        y_delta=-(y2 - y1) / 15,
        graybox_y=(y0, y2),
        ax=ax,
    )

    for partial_alignment in eval.partial_alignments:
        draw_partial_alignment(
            partial_alignment,
            ax=ax,
            show_processed_time=False,
            show_send_time=False,
        )
    
    real_times = np.linspace(
        eval.start_timestamp, eval.finish_timestamp, num=3000
    )
    sent_times = [
        get_audio_seconds_sent(t, eval.input_chunks) for t in real_times
    ]
    processed_times = [
        get_audio_seconds_processed(t, eval.output_chunks) for t in real_times
    ]
    ax.plot(processed_times, real_times, color='darkgreen', zorder=1) # type: ignore
    ax.plot(sent_times, real_times, color='grey', zorder=0) # type: ignore
        
    ax.set_xlabel('Audio time') # type: ignore
    ax.set_ylabel('Real time') # type: ignore


def visualize_history(
    input_chunks: list[InputChunk],
    output_chunks: list[OutputChunk] | None = None,
    ax: plt.Axes | None = None,
):
    """Visualize the history of sending and receiving chunks.
    
    See more details and examples in the user guide:
    :doc:`/guide_streaming_evaluation`.
    """
    
    ax = ax or plt.gca()
        
    plot_t_shift = input_chunks[0].put_timestamp

    plot_y_pos = 0
    for input_chunk in input_chunks:
        ax.scatter( # type: ignore
            [input_chunk.get_timestamp - plot_t_shift, \
                input_chunk.put_timestamp - plot_t_shift],
            [plot_y_pos, plot_y_pos],
            c=['b', 'g'],
            s=30,
            marker='|',
        )
        plot_y_pos += 1

    if output_chunks is not None:
        plot_y_pos = 0
        for output_chunk in output_chunks:
            ax.axvline( # type: ignore
                output_chunk.put_timestamp - plot_t_shift,
                c='orange',
                alpha=1 if output_chunk.data is Signal.FINISH else 0.5,
                ls='dashed' if output_chunk.data is Signal.FINISH else 'solid',
                zorder=-1,
            )
        
    ax.set_xlabel('Real time') # type: ignore
    ax.set_ylabel('Chunk index') # type: ignore


def streaming_error_vs_latency_histogram(
    evals: list[StreamingEvaluationResults],
    ax: plt.Axes | None = None,
    max_latency: float = 10,
):
    """Summarizes error percentage versus latency in a historgram, given
    evaluations for multiple samples.
    
    See more details and examples in the user guide:
    :doc:`/guide_streaming_evaluation`.
    """
    
    ax = ax or plt.gca()
    
    error_positions = sum([
        pa.get_error_positions()
        for eval in evals
        for pa in eval.partial_alignments
    ], [])  # pyright: ignore[reportUnknownArgumentType]
    
    counts: dict[Literal['correct', 'error', 'not_yet'], INTS] = {}
    
    bins = np.linspace(0, 10, num=31).round(2).tolist()
    bins = [b for b in bins if b <= max_latency]
    bins += [np.inf]

    for status, pos_status in [
        ('correct', ['correct']),
        ('error', ['deletion', 'replacement', 'insertion']),
        ('not_yet', ['not_yet']),
    ]:
        counts[status] = np.histogram( # type: ignore
            [
                # todo 2 options: sent or processed time
                x.sent_time - x.center_time
                for x in error_positions
                if x.status in pos_status
            ],
            bins=bins
        )[0]

    total_counts = sum(counts.values())

    ratios = {status: c / total_counts for status, c in counts.items()}

    xrange = range(len(bins) - 1)
    ax.bar(xrange, height=ratios['correct'], color='green') # type: ignore
    ax.bar(xrange, height=ratios['error'], # type: ignore
           bottom=ratios['correct'], color='C1')
    ax.bar(xrange, height=ratios['not_yet'], # type: ignore
           bottom=ratios['correct'] + ratios['error'], color='gray')
    ax.set_xticks(xrange) # type: ignore
    ax.set_xticklabels( # type: ignore
        [
            f'{a:g}-{b:g}' if np.isfinite(b) else f'{a:g}-inf'
            for a, b in pairwise(bins)
        ],
        rotation=90,
    )


def latency_plot(
    evals: list[StreamingEvaluationResults],
    ax: plt.Axes | None = None,
):
    """Summarizes latencies, given evaluations for multiple samples.
    
    See more details and examples in the user guide:
    :doc:`/guide_streaming_evaluation`.
    """
    
    ax = ax or plt.gca()
    
    for eval in evals:
        sent = np.array([
            pa.audio_seconds_sent for pa in eval.partial_alignments
        ])
        processed = np.array([
            pa.audio_seconds_processed for pa in eval.partial_alignments
        ])
        ax.plot(processed, sent, alpha=0.3, lw=1, color='C0') # type: ignore

    ax.set_xlabel('Audio time processed, sec') # type: ignore
    ax.set_ylabel('Audio time sent, sec') # type: ignore


def show_last_alignments(
    evals: list[StreamingEvaluationResults],
    ax: plt.Axes | None = None,
):
    """Visualizes the last alignments (finalized transcriptions),
    given evaluations for multiple samples.

    Sorts evals by audio length before drawing.
    
    See more details and examples in the user guide:
    :doc:`/guide_streaming_evaluation`.
    """
    
    ax = ax or plt.gca()
    
    last_partial_alignments = [eval.partial_alignments[-1] for eval in evals]
    
    for i, al in enumerate(
        sorted(last_partial_alignments, key=lambda pa: pa.audio_seconds_sent)
    ):
        draw_partial_alignment(al, ax=ax, override_y_pos=i)
        
    ax.set_xlabel('Audio time') # type: ignore
    ax.set_ylabel('Sample index') # type: ignore