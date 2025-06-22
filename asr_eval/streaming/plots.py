from typing import Literal, cast
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from .model import InputChunk, OutputChunk, Signal
from ..align.data import Token
from .evaluation import PartialAlignment, RecordingStreamingEvaluation
from ..utils.utils import N


def draw_partial_alignment(
    partial_alignment: PartialAlignment,
    override_y_pos: float | None = None,
    ax: plt.Axes | None = None,
):
    ax = ax or plt.gca()
    y_pos = override_y_pos if override_y_pos is not None else partial_alignment.at_time
    for pos in partial_alignment.get_error_positions():
        if pos.status == 'insertion':
            ax.scatter( # type: ignore
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
            ax.plot([pos.start_time, pos.end_time], [y_pos, y_pos], color=color) # type: ignore
    if partial_alignment.audio_seconds_processed is not None:
        ax.scatter( # type: ignore
            [partial_alignment.audio_seconds_processed], [y_pos], # type: ignore
            s=30, zorder=2, color='red', marker='|'
        )
    ax.scatter( # type: ignore
        [partial_alignment.audio_seconds_sent], [y_pos], # type: ignore
        s=20, zorder=2, color='red', marker='.'
    )


def partial_alignments_plot(
    eval: RecordingStreamingEvaluation,
    ax: plt.Axes | None = None,
):
    """
    Displays a partial alignment diagram, given the result of `get_partial_alignments()`.
    
    TODO merge with `visualize_history()`
    """
    ax = ax or plt.gca()

    # word timings
    for token in cast(list[Token], eval.recording.transcription_words):
        assert isinstance(token, Token)
        assert token.is_timed
        ax.fill_between( # type: ignore
            [token.start_time, token.end_time],
            [eval.start_timestamp, eval.start_timestamp],
            [eval.finish_timestamp, eval.finish_timestamp],
            color='#eeeeee',
            zorder=-1
        )
        ax.text( # type: ignore
            (token.start_time + token.end_time) / 2,
            eval.start_timestamp,
            ' ' + str(token.value),
            fontsize=10,
            rotation=90,
            ha='center',
            va='bottom',
        )

    for partial_alignment in eval.partial_alignments:
        draw_partial_alignment(partial_alignment, ax=ax)
        
    ax.set_xlabel('Audio time') # type: ignore
    ax.set_ylabel('Real time') # type: ignore


def visualize_history(
    input_chunks: list[InputChunk],
    output_chunks: list[OutputChunk] | None = None,
    ax: plt.Axes | None = None,
):
    """
    Visualize the history of sending and receiving chunks.
    
    TODO merge with `partial_alignment_diagram()`
    """
    ax = ax or plt.gca()
        
    plot_t_shift = cast(float, input_chunks[0].put_timestamp)

    plot_y_pos = 0
    for input_chunk in input_chunks:
        ax.scatter( # type: ignore
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
            ax.axvline( # type: ignore
                N(output_chunk.put_timestamp) - plot_t_shift,
                c='orange',
                alpha=1 if output_chunk.data is Signal.FINISH else 0.5,
                ls='dashed' if output_chunk.data is Signal.FINISH else 'solid',
                zorder=-1,
            )
        
    ax.set_xlabel('Real time') # type: ignore
    ax.set_ylabel('Chunk index') # type: ignore
    # extend_lims(plt.gca(), dxmin=-0.1, dxmax=0.1, dymin=-1, dymax=1)


def streaming_error_vs_latency_histogram(
    evals: list[RecordingStreamingEvaluation],
    ax: plt.Axes | None = None,
):
    ax = ax or plt.gca()
    
    error_positions = sum([
        pa.get_error_positions()
        for eval in evals
        for pa in eval.partial_alignments
    ], [])  # pyright: ignore[reportUnknownArgumentType]
    
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

    xrange = range(len(bins) - 1)
    ax.bar(xrange, height=ratios['correct']) # type: ignore
    ax.bar(xrange, height=ratios['error'], bottom=ratios['correct']) # type: ignore
    ax.bar(xrange, height=ratios['not_yet'], bottom=ratios['correct'] + ratios['error']) # type: ignore
    ax.set_xticks(xrange) # type: ignore
    ax.set_xticklabels([f'{a:g}-{b:g}' for a, b in pairwise(bins)], rotation=90) # type: ignore


def latency_plot(
    evals: list[RecordingStreamingEvaluation],
    ax: plt.Axes | None = None,
):
    ax = ax or plt.gca()
    
    for eval in evals:
        sent = np.array([pa.audio_seconds_sent for pa in eval.partial_alignments])
        processed = np.array([pa.audio_seconds_processed for pa in eval.partial_alignments])
        ax.plot(processed, sent, alpha=0.5, lw=3, color='C0') # type: ignore

    ax.set_xlabel('Audio time processed, sec') # type: ignore
    ax.set_ylabel('Audio time sent, sec') # type: ignore


def show_last_alignments(
    evals: list[RecordingStreamingEvaluation],
    ax: plt.Axes | None = None,
):
    ax = ax or plt.gca()
    
    last_partial_alignments = [eval.partial_alignments[-1] for eval in evals]
    
    for i, al in enumerate(sorted(last_partial_alignments, key=lambda pa: pa.audio_seconds_sent)):
        draw_partial_alignment(al, ax=ax, override_y_pos=i)
        
    ax.set_xlabel('Audio time') # type: ignore
    ax.set_ylabel('Sample index') # type: ignore