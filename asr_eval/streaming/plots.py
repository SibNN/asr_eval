from typing import cast

import numpy as np
import matplotlib.pyplot as plt

from .model import InputChunk, OutputChunk, Signal
from ..align.data import Token
from .evaluation import PartialAlignment
from ..utils import N


def partial_alignment_diagram(
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
    last_end_time = 0
    for partial_alignment in partial_alignments:
        y_pos = partial_alignment.at_time
        for match in partial_alignment.alignment.matches:
            if len(match.true) == 0:
                plt.scatter([last_end_time], [y_pos], color='black', s=10, zorder=2) # type: ignore
            else:
                assert len(match.true) == 1
                last_end_time = match.true[0].end_time

                skip = False
                if match.status == 'correct':
                    color = 'green'
                elif match.status == 'replacement':
                    color = 'red'
                else:
                    assert match.status == 'deletion'
                    color = 'gray'
                    # skip = True
                
                if not skip:
                    for token in match.true:
                        plt.plot([token.start_time, token.end_time], [y_pos, y_pos], color=color) # type: ignore
        if partial_alignment.audio_seconds_processed is not None:
            plt.scatter( # type: ignore
                [partial_alignment.audio_seconds_processed], [y_pos], # type: ignore
                s=20, zorder=2, color='gray', marker='|'
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