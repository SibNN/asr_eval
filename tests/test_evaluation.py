from pathlib import Path

import pytest
import librosa

from asr_eval.align.timings import fill_word_timings_inplace
from asr_eval.align.parsing import DEFAULT_PARSER
from asr_eval.align.plots import draw_timed_transcription
from asr_eval.models.gigaam_wrapper import GigaAMShortformCTC
from asr_eval.models.vosk_streaming_wrapper import VoskStreaming
from asr_eval.streaming.caller import receive_transcription
from asr_eval.streaming.evaluation import (
    evaluate_streaming, remap_time, make_sender
)
from asr_eval.streaming.plots import (
    partial_alignments_plot,
    visualize_history,
    streaming_error_vs_latency_histogram,
    latency_plot,
    show_last_alignments,
)
from asr_eval.utils.types import FLOATS


@pytest.mark.skip(reason='todo decide how to test optional dependencies')
@pytest.mark.filterwarnings(
    'ignore::FutureWarning:', 'ignore::DeprecationWarning:'
)
def test_evaluation():
    waveform: FLOATS = librosa.load('tests/testdata/formula1.mp3', sr=16000)[0] # type: ignore
    waveform += waveform[::-1] / 4  # add speech-like noise
    text = Path('tests/testdata/formula1.txt').read_text()
    
    # parse multivariant transcription
    ground_truth = DEFAULT_PARSER.parse_transcription(text)

    # determine word timings
    model = GigaAMShortformCTC('v2')
    fill_word_timings_inplace(model, waveform, ground_truth)
    
    # plot should not raise an error
    ground_truth.colorize()
    draw_timed_transcription(ground_truth, y_delta=-3)
    
    # transcribing
    asr = VoskStreaming(model_name='vosk-model-ru-0.42', chunk_length_sec=1)
    asr.start_thread()
    cutoffs, sender = make_sender(waveform, asr)
    sender.start_sending(without_delays=True)
    output_chunks = list(receive_transcription(asr=asr, id=sender.id))
    input_chunks = sender.join_and_get_history()
    input_chunks, output_chunks = remap_time(cutoffs, input_chunks, output_chunks)
    eval = evaluate_streaming(
        ground_truth,
        waveform,
        cutoffs,
        input_chunks,
        output_chunks,
        partial_alignment_interval=1.5,
    )
    asr.stop_thread()
    
    # regression testing for the outputs
    assert len(eval.input_chunks) == 81
    print([str(t.value) for t in eval.partial_alignments[-1].pred.blocks])
    assert [str(t.value) for t in eval.partial_alignments[-1].pred.blocks] == (
        ['седьмого', 'восьмого', 'мая', 'по', 'эру', 'торика', 'прошел', 'шестнадцатый', 'этап', 'формулы',
         'один', 'с', 'фондом', 'сто', 'тысяч', 'долларов', 'победителем', 'стал', 'гонщик', 'мерседеса']
    )
    
    # unstable:
    # assert [pa.alignment.score.n_word_errors for pa in eval.partial_alignments] == [0, 1, 3, 3, 5, 5, 7, 7, 4, 6, 7, 6]
    # assert [''.join([x.status[0] for x in pa.get_error_positions()]) for pa in eval.partial_alignments] == [
    #     # c - correct, n - not_yet, d - deletion, i - insertion, r - replacement
    #     '',
    #     'n',
    #     'cnnn',
    #     'cccnnn',
    #     'cccdrdrr',
    #     'cccrrrcnn',
    #     'cccrrrccrcnnn',
    #     'cccrrrccrccdrn',
    #     'cccrrrccrccccccc',
    #     'cccrrrccrcccccccnn',
    #     'cccrrrccrcccccccccnnn',
    #     'cccrrrccrcccccccccccnn',
    # ]
    
    # plots should not raise an error
    partial_alignments_plot(eval)
    visualize_history(eval.input_chunks, eval.output_chunks)
    streaming_error_vs_latency_histogram([eval])
    latency_plot([eval])
    show_last_alignments([eval])
    