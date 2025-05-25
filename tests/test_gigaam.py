import typing

import pytest
import gigaam # pyright: ignore[reportMissingTypeStubs]
from gigaam.model import GigaAMASR # pyright: ignore[reportMissingTypeStubs]
import torch
import librosa
import numpy as np
import numpy.typing as npt

from asr_eval.models.gigaam import transcribe_with_gigaam_ctc, decode_each_token
from asr_eval.ctc_utils import ctc_mapping


@pytest.fixture
def model() -> GigaAMASR:
    return typing.cast(GigaAMASR, gigaam.load_model('ctc', device='cpu'))

@pytest.fixture
def model_cuda() -> GigaAMASR:
    return typing.cast(GigaAMASR, gigaam.load_model('ctc', device='cuda'))

@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::UserWarning:', 'ignore::DeprecationWarning:')
@pytest.mark.parametrize('model', ['model', 'model_cuda'], indirect=True)
def test_prepare_audio_for_gigaam(model: GigaAMASR):
    '''
    Check that we can prepare audio manually instead of GigaAM.prepare_wav(path) and get the same result.
    This is useful because prepare_wav accepts path, but typically we have a numpy waveform instead.
    '''
    audio_path = 'tests/testdata/podlodka_test_0.wav'
    
    # load using gigaam
    waveform_torch_from_giaam, length = model.prepare_wav(audio_path)

    # load using librosa
    waveform: npt.NDArray[np.floating]
    waveform, _ = librosa.load(audio_path, sr=16_000) # type: ignore
    waveform_torch_from_librosa = (
        torch.tensor(waveform, dtype=model._dtype).to(model._device).unsqueeze(0) # pyright: ignore[reportPrivateUsage]
    )
    length_from_librosa = (
        torch.tensor([waveform_torch_from_librosa.shape[1]]).to(model._device) # pyright: ignore[reportPrivateUsage]
    )

    assert torch.all(waveform_torch_from_giaam == waveform_torch_from_librosa)
    assert waveform_torch_from_giaam.shape[1] == length[0]
    assert waveform_torch_from_librosa.shape[1] == length[0]
    assert length.dtype == length_from_librosa.dtype
    assert torch.all(length == length_from_librosa)

@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::UserWarning:')
@pytest.mark.parametrize('model', ['model', 'model_cuda'], indirect=True)
def test_giggam(model: GigaAMASR):
    audio_path1 = 'tests/testdata/podlodka_test_0.wav'
    audio_path2 = 'tests/testdata/vosk.wav'
    
    # expected gigaam prediction
    expected_text1 = 'и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно'
    expected_text2 = 'вон зироу зироу зироу вон наноу ту вон оу зироу вон эйт сироу три'
    
    assert model.transcribe(audio_path1) == expected_text1
    assert model.transcribe(audio_path2) == expected_text2

    waveforms: list[npt.NDArray[np.floating]] = [
        librosa.load(audio_path1, sr=16_000)[0], # type: ignore
        librosa.load(audio_path2, sr=16_000)[0], # type: ignore
    ]
    output1, output2 = transcribe_with_gigaam_ctc(model, waveforms)
    assert output1.text == expected_text1
    assert output2.text == expected_text2
    
    output1.visualize(model, waveforms[0])
    output2.visualize(model, waveforms[1])
    
    symbols = decode_each_token(model, output1.labels)
    assert ''.join(ctc_mapping(symbols)) == expected_text1
    
    symbols = decode_each_token(model, output2.labels)
    assert ''.join(ctc_mapping(symbols)) == expected_text2