import typing

import pytest
import librosa
import gigaam # pyright: ignore[reportMissingTypeStubs]
from gigaam.model import GigaAMASR # pyright: ignore[reportMissingTypeStubs]

from asr_eval.models.gigaam import transcribe_with_gigaam_ctc
from asr_eval.force_alignment import force_alignment, plot_alignments, plot_trellis_with_path


@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::DeprecationWarning:')
def test_force_alignment():
    waveform, _ = librosa.load('tests/testdata/podlodka_test_0.wav', sr=16_000) # type: ignore
    true_text = 'и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно'
    
    model = typing.cast(GigaAMASR, gigaam.load_model('ctc', device='cpu'))
    tokens = [model.decoding.tokenizer.vocab.index(x) for x in true_text]
    output = transcribe_with_gigaam_ctc(model, [waveform])[0]
    
    segments, path, trellis = force_alignment(output.log_probs, tokens, model.decoding.blank_id)
    
    assert ''.join([model.decoding.tokenizer.vocab[s.token] for s in segments]) == true_text
    for s in segments:
        assert s.end > s.start
        assert s.score <= 1
    for s1, s2 in zip(segments, segments[1:]):
        assert s2.start == s1.end
    
    plot_alignments(segments,trellis, model.decoding.tokenizer.vocab)
    plot_trellis_with_path(trellis, path)