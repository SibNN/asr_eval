import pytest
import torch
import librosa

from asr_eval.models.gigaam_wrapper import GigaAMShortformCTC
from asr_eval.utils.types import FLOATS


@pytest.fixture
def model() -> GigaAMShortformCTC:
    return GigaAMShortformCTC(device='cpu')

@pytest.fixture
def model_cuda() -> GigaAMShortformCTC:
    return GigaAMShortformCTC(device='cuda')

@pytest.mark.skip(reason='todo decide how to test optional dependencies')
@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::UserWarning:', 'ignore::DeprecationWarning:')
@pytest.mark.parametrize('model', ['model', 'model_cuda'], indirect=True)
def test_prepare_audio_for_gigaam(model: GigaAMShortformCTC):
    '''
    Check that we can prepare audio manually instead of GigaAM.prepare_wav(path) and get the same result.
    This is useful because prepare_wav accepts path, but typically we have a numpy waveform instead.
    '''
    audio_path = 'tests/testdata/podlodka_test_0.wav'
    
    # load using gigaam
    waveform_torch_from_giaam, length = model.model.prepare_wav(audio_path)

    # load using librosa
    waveform: FLOATS = librosa.load(audio_path, sr=16_000)[0] # type: ignore
    waveform_torch_from_librosa = (
        torch.tensor(waveform, dtype=model.model._dtype).to(model.model._device).unsqueeze(0) # pyright: ignore[reportPrivateUsage]
    )
    length_from_librosa = (
        torch.tensor([waveform_torch_from_librosa.shape[1]]).to(model.model._device) # pyright: ignore[reportPrivateUsage]
    )

    assert torch.all(waveform_torch_from_giaam == waveform_torch_from_librosa)
    assert waveform_torch_from_giaam.shape[1] == length[0]
    assert waveform_torch_from_librosa.shape[1] == length[0]
    assert length.dtype == length_from_librosa.dtype
    assert torch.all(length == length_from_librosa)

@pytest.mark.skip(reason='todo decide how to test optional dependencies')
@pytest.mark.filterwarnings('ignore::FutureWarning:', 'ignore::UserWarning:')
@pytest.mark.parametrize('model', ['model', 'model_cuda'], indirect=True)
def test_giggam(model: GigaAMShortformCTC):
    audio_path1 = 'tests/testdata/podlodka_test_0.wav'
    audio_path2 = 'tests/testdata/vosk.wav'
    
    # expected gigaam prediction
    expected_text1 = 'и поэтому использовать их в повседневности не получается мы вынуждены поступать зачастую интуитивно'
    expected_text2 = 'вон зироу зироу зироу вон наноу ту вон оу зироу вон эйт сироу три'
    
    assert model.model.transcribe(audio_path1) == expected_text1
    assert model.model.transcribe(audio_path2) == expected_text2

    waveforms: list[FLOATS] = [
        librosa.load(audio_path1, sr=16_000)[0], # type: ignore
        librosa.load(audio_path2, sr=16_000)[0], # type: ignore
    ]
    text1 = model.transcribe(waveforms[0])
    text2 = model.transcribe(waveforms[1])
    assert text1 == expected_text1
    assert text2 == expected_text2

    # for log_probs, expected_text in [
    #     (output1.log_probs, expected_text1),
    #     (output2.log_probs, expected_text2),
    # ]:
    #     tokens = ctc_mapping(cast(list[int], log_probs.argmax(axis=1).tolist()), blank=model.decoding.blank_id)
    #     assert gigaam_decode(model, tokens) == expected_text