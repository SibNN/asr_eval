import typing

import pytest
import librosa
import gigaam # pyright: ignore[reportMissingTypeStubs]
from gigaam.model import GigaAMASR # pyright: ignore[reportMissingTypeStubs]
import numpy as np

from asr_eval.models.gigaam import transcribe_with_gigaam_ctc
from asr_eval.ctc_utils import ctc_mapping, recursion_ctc_forced_alignment, torch_ctc_forced_alignment

def test_ctc_mapping():
    assert ctc_mapping(list('____')) == []
    assert ctc_mapping(list('aaaaa')) == ['a']
    
    x = list('_________дджжой   иссто__ч_ни__ки_________   _иссто_ри__и')
    assert ctc_mapping(x) == list('джой источники истории')
    

@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::DeprecationWarning:')
def test_forced_alignment():
    waveform, _ = librosa.load('tests/testdata/podlodka_test_0.wav', sr=16_000) # type: ignore
    text = 'и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно'
    
    model = typing.cast(GigaAMASR, gigaam.load_model('ctc', device='cpu'))
    tokens = [model.decoding.tokenizer.vocab.index(x) for x in text]
    output = transcribe_with_gigaam_ctc(model, [waveform])[0]
    
    # use model.decoding.blank_id as blank token
    
    tokens1, p1 = recursion_ctc_forced_alignment(output.log_probs, tokens, model.decoding.blank_id)
    tokens2, p2 = torch_ctc_forced_alignment(output.log_probs, tokens, model.decoding.blank_id)
    
    assert np.allclose(p1, p2)
    assert tokens1 == tokens2
    
    decoded_each_token = [
        model.decoding.tokenizer.vocab[x]
        if x != model.decoding.blank_id
        else '_'
        for x in tokens1
    ]
    
    assert ''.join(ctc_mapping(decoded_each_token)) == text
    
    # use 0 as blank token
    
    for fa in [torch_ctc_forced_alignment, recursion_ctc_forced_alignment]: # type: ignore
    
        n_tokens = output.log_probs.shape[1]
        tokens3, p3 = fa( # type: ignore
            np.ascontiguousarray(output.log_probs[:, [n_tokens - 1] + list(range(n_tokens - 1))]),
            [(t + 1 if t != n_tokens - 1 else 0) for t in tokens],
            0,
        )
        assert np.allclose(p1, p3) # type: ignore
        assert tokens1 == [(t - 1 if t != 0 else n_tokens - 1) for t in tokens3]