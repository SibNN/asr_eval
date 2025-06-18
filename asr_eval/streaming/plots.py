from typing import Literal, cast
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from asr_eval.data import Recording

from .model import InputChunk, OutputChunk, Signal
from ..align.data import Token
from .evaluation import PartialAlignment, StreamingASRErrorPosition
from ..utils import N


def partial_alignment_plot(
    partial_alignments: list[PartialAlignment],
    true_words_timed: list[Token],
    start_real_time: float,
    end_real_time: float,
    # audio_len: float,
    figsize: tuple[float, float] = (15, 15),
):
    """
    Displays a partial alignment diagram, given the result of `get_partial_alignments()`.
    
    TODO merge with `visualize_history()`
    """
    plt.figure(figsize=figsize) # type: ignore

    # main lines
    # plt.plot([0, audio_len], [0, audio_len], color='lightgray') # type: ignore
    # plt.plot([0, audio_len], [0, 0], color='lightgray') # type: ignore

    # word timings
    for token in true_words_timed:
        assert not np.isnan(token.start_time)
        assert not np.isnan(token.end_time)
        plt.fill_between( # type: ignore
            [token.start_time, token.end_time],
            [start_real_time, start_real_time],
            [end_real_time, end_real_time],
            color='#eeeeee',
            zorder=-1
        )
        plt.text( # type: ignore
            (token.start_time + token.end_time) / 2,
            start_real_time,
            ' ' + str(token.value),
            fontsize=10,
            rotation=90,
            ha='center',
            va='bottom',
        )

    # partial alignments
    # last_end_time = 0
    for partial_alignment in partial_alignments:
        y_pos = partial_alignment.at_time
        for pos in partial_alignment.get_error_positions():
            if pos.status == 'insertion':
                plt.scatter( # type: ignore
                    [(pos.start_time + pos.end_time) / 2], [y_pos], color='darkred', s=10, zorder=3
                )
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
                plt.plot([pos.start_time, pos.end_time], [y_pos, y_pos], color=color) # type: ignore
        if partial_alignment.audio_seconds_processed is not None:
            plt.scatter( # type: ignore
                [partial_alignment.audio_seconds_processed], [y_pos], # type: ignore
                s=30, zorder=2, color='red', marker='|'
            )
        plt.scatter( # type: ignore
            [partial_alignment.audio_seconds_sent], [y_pos], # type: ignore
            s=20, zorder=2, color='red', marker='.'
        )

    plt.show() # type: ignore


def visualize_history(
    input_chunks: list[InputChunk],
    output_chunks: list[OutputChunk] | None = None,
):
    """
    Visualize the history of sending and receiving chunks.
    
    TODO merge with `partial_alignment_diagram()`
    """    
    plt.figure(figsize=(6, 6)) # type: ignore
        
    plot_t_shift = cast(float, input_chunks[0].put_timestamp)

    plot_y_pos = 0
    for input_chunk in input_chunks:
        plt.scatter( # type: ignore
            [N(input_chunk.get_timestamp) - plot_t_shift, N(input_chunk.put_timestamp) - plot_t_shift],
            [plot_y_pos, plot_y_pos],
            c=['b', 'g'],
            s=30,
            marker='|',
        )
        plot_y_pos += 1

    if output_chunks is not None:
        plot_y_pos = 0
        for output_chunk in output_chunks:
            plt.axvline( # type: ignore
                N(output_chunk.put_timestamp) - plot_t_shift,
                c='orange',
                alpha=1 if output_chunk.data is Signal.FINISH else 0.5,
                ls='dashed' if output_chunk.data is Signal.FINISH else 'solid',
                zorder=-1,
            )
        
    # extend_lims(plt.gca(), dxmin=-0.1, dxmax=0.1, dymin=-1, dymax=1)
    plt.show() # type: ignore


def streaming_error_vs_latency_histogram(
    error_positions: list[StreamingASRErrorPosition],
):
    counts: dict[Literal['correct', 'error', 'not_yet'], npt.NDArray[np.int64]] = {}

    # bins = [0, 1, 2, 3, 5, 10, 1000]
    bins = np.linspace(0, 10, num=31).round(2).tolist() + [1000]
    for status, pos_status in [
        ('correct', ['correct']),
        ('error', ['deletion', 'replacement', 'insertion']),
        ('not_yet', ['not_yet']),
    ]:
        counts[status] = np.histogram( # type: ignore
            [x.time_delta for x in error_positions if x.status in pos_status],
            bins=bins
        )[0]

    total_counts = sum(counts.values())

    ratios = {status: c / total_counts for status, c in counts.items()}

    plt.figure(figsize=(10, 3)) # type: ignore
    xrange = range(len(bins) - 1)
    plt.bar(xrange, height=ratios['correct']) # type: ignore
    plt.bar(xrange, height=ratios['error'], bottom=ratios['correct']) # type: ignore
    plt.bar(xrange, height=ratios['not_yet'], bottom=ratios['correct'] + ratios['error']) # type: ignore
    plt.gca().set_xticks(xrange) # type: ignore
    plt.gca().set_xticklabels([f'{a:g}-{b:g}' for a, b in pairwise(bins)], rotation=90) # type: ignore
    plt.show() # type: ignore


def latency_plot(samples: list[Recording]):
    plt.figure(figsize=(10, 5)) # type: ignore
    
    for recording in samples:
        sent = np.array([pa.audio_seconds_sent for pa in N(N(recording.evals).partial_alignments)])
        processed = np.array([pa.audio_seconds_processed for pa in N(N(recording.evals).partial_alignments)])
        plt.plot(sent, sent - processed, alpha=0.5, lw=3, color='C0') # type: ignore

    plt.xlabel('Sent, sec') # type: ignore
    plt.ylabel('Processed, sec') # type: ignore
    plt.show() # type: ignore