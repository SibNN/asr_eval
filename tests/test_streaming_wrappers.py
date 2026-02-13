from pathlib import Path

import librosa
import pytest

from asr_eval.align.parsing import DEFAULT_PARSER
from asr_eval.align.timings import fill_word_timings_inplace
# from asr_eval.bench.recording import Recording
from asr_eval.models.gigaam_wrapper import GigaAMShortformCTC
from asr_eval.models.vosk_streaming_wrapper import VoskStreaming
from asr_eval.streaming.caller import receive_transcription
from asr_eval.streaming.evaluation import evaluate_streaming, remap_time, make_sender
from asr_eval.streaming.wrappers import StreamingToOffline, OfflineToStreaming
from asr_eval.streaming.plots import partial_alignments_plot
from asr_eval.streaming.model import TranscriptionChunk
from asr_eval.utils.types import FLOATS


@pytest.mark.skip(reason='todo decide how to test optional dependencies')
@pytest.mark.filterwarnings('ignore::DeprecationWarning:')
def test_streaming_to_offline():
    waveform: FLOATS = librosa.load('tests/testdata/vosk.wav', sr=16_000)[0] # type: ignore

    model = StreamingToOffline(VoskStreaming())
    assert model.transcribe(waveform) == 'one zero zero zero one nah no to i know zero one eight zero three'
    
    model.streaming_model.stop_thread()
    

@pytest.mark.skip(reason='todo decide how to test optional dependencies')
def test_offline_to_streaming():
    waveform: FLOATS = librosa.load('tests/testdata/formula1.mp3', sr=16_000)[0] # type: ignore
    transcription = DEFAULT_PARSER.parse_transcription(Path('tests/testdata/formula1.txt').read_text())

    gigaam = GigaAMShortformCTC('v2')
    fill_word_timings_inplace(gigaam, waveform, transcription)

    model = OfflineToStreaming(gigaam)
    model.start_thread()
    cutoffs, sender = make_sender(waveform, model)
    sender.start_sending()
    output_chunks = list(receive_transcription(asr=model, id=sender.id))
    input_chunks = sender.join_and_get_history()
    input_chunks, output_chunks = remap_time(cutoffs, input_chunks, output_chunks)
    eval = evaluate_streaming(
        transcription,
        waveform,
        cutoffs,
        input_chunks,
        output_chunks,
    )
    partial_alignments_plot(eval)
    
    assert TranscriptionChunk.join(eval.output_chunks) == (
        'седьмого восьмого мая в пуэрто рико прошел шестнадцатый этап формулы один'
        ' с фондом сто тысяч долларов победителем стал гонщик мерседеса джордж рассел'
    )

    model.stop_thread()
