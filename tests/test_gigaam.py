import typing

import pytest
import gigaam # pyright: ignore[reportMissingTypeStubs]
from gigaam.model import GigaAMASR # pyright: ignore[reportMissingTypeStubs]
import torch
import librosa
import numpy as np
import numpy.typing as npt

from asr_eval.models.gigaam import transcribe_with_gigaam_ctc


@pytest.fixture
def model() -> GigaAMASR:
    return typing.cast(GigaAMASR, gigaam.load_model('ctc', device='cpu'))

@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::UserWarning:', 'ignore::DeprecationWarning:')
def test_prepare_audio_for_gigaam(model: GigaAMASR):
    '''
    Check that we can prepare audio manually instead of GigaAM.prepare_wav(path) and get the same result.
    This is useful because prepare_wav accepts path, but typically we have a numpy waveform instead.
    '''
    audio_path = 'tests/testdata/podlodka_test_0.wav'
    
    # load using gigaam
    waveform_torch_from_giaam, length = model.prepare_wav(audio_path)

    # load using librosa
    waveform: npt.NDArray[np.float64]
    waveform, _ = librosa.load(audio_path, sr=16_000) # type: ignore
    waveform_torch_from_librosa = torch.tensor(waveform, dtype=model._dtype).unsqueeze(0) # pyright: ignore[reportPrivateUsage]
    length_from_librosa = torch.tensor([waveform_torch_from_librosa.shape[1]])

    assert torch.all(waveform_torch_from_giaam == waveform_torch_from_librosa)
    assert waveform_torch_from_giaam.shape[1] == length[0]
    assert waveform_torch_from_librosa.shape[1] == length[0]
    assert length.dtype == length_from_librosa.dtype
    assert torch.all(length == length_from_librosa)

@pytest.mark.filterwarnings('ignore::FutureWarning:')
def test_giggam(model: GigaAMASR):
    audio_path = 'tests/testdata/podlodka_test_0.wav'
    
    # expected gigaam prediction
    expected_text = 'и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно'
    
    assert model.transcribe(audio_path) == expected_text

    waveform: npt.NDArray[np.float64]
    waveform, _ = librosa.load(audio_path, sr=16_000) # type: ignore
    output = transcribe_with_gigaam_ctc(model, waveform)
    assert output.text == expected_text