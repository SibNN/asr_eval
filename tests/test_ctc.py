import typing

import pytest
import librosa
import gigaam
from gigaam.model import GigaAMASR
import numpy as np

from asr_eval.models.gigaam_wrapper import transcribe_with_gigaam_ctc, gigaam_decode
from asr_eval.ctc.base import ctc_mapping
from asr_eval.ctc.forced_alignment import recursion_forced_alignment, forced_alignment
from asr_eval.utils.types import FLOATS

def test_ctc_mapping():
    assert ctc_mapping(list('____'), blank='_') == []
    assert ctc_mapping(list('aaaaa'), blank='_') == ['a']
    
    x = list('_________дджжой   иссто__ч_ни__ки_________   _иссто_ри__и')
    assert ctc_mapping(x, blank='_') == list('джой источники истории')
    

@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::DeprecationWarning:')
def test_forced_alignment():
    waveform: FLOATS = librosa.load('tests/testdata/podlodka_test_0.wav', sr=16_000)[0] # type: ignore
    text = 'и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно'
    
    model = typing.cast(GigaAMASR, gigaam.load_model('ctc', device='cpu'))
    tokens = [model.decoding.tokenizer.vocab.index(x) for x in text]
    output = transcribe_with_gigaam_ctc(model, [waveform])[0]
    
    # use model.decoding.blank_id as blank token
    
    tokens1, p1 = recursion_forced_alignment(output.log_probs, tokens, model.decoding.blank_id)
    tokens2, p2, _ = forced_alignment(output.log_probs, tokens, model.decoding.blank_id)
    
    assert np.allclose(p1, p2)
    assert tokens1 == tokens2
    
    assert gigaam_decode(model, ctc_mapping(tokens1, blank=model.decoding.blank_id)) == text
    
    # use 0 as blank token
    
    for fa in [forced_alignment, recursion_forced_alignment]: # type: ignore
    
        n_tokens = output.log_probs.shape[1]
        fa_result = fa( # type: ignore
            np.ascontiguousarray(output.log_probs[:, [n_tokens - 1] + list(range(n_tokens - 1))]),
            [(t + 1 if t != n_tokens - 1 else 0) for t in tokens],
            0,
        )
        tokens3, p3 = fa_result[:2]
        assert np.allclose(p1, p3) # type: ignore
        assert tokens1 == [(t - 1 if t != 0 else n_tokens - 1) for t in tokens3] # type: ignore