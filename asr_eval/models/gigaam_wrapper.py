from __future__ import annotations
from abc import ABC, abstractmethod
import shutil
from typing import TYPE_CHECKING, Literal, cast, override
import warnings

if TYPE_CHECKING:
    from gigaam.model import GigaAMASR, GigaAM, GigaAMEmo
    import torch

from asr_eval.models.base.interfaces import CTC, Transcriber
from asr_eval.utils.types import FLOATS


__all__ = [
    'GigaAMShortformBase',
    'GigaAMShortformRNNT',
    'GigaAMShortformCTC',
]


SAMPLING_RATE = 16000
FREQ = 25  # GigaAM2 encoder outputs per second


class GigaAMShortformBase(Transcriber, ABC):
    '''
    An abstract class for GigaAM model, either CTC or RNNT.
    
    Implementations:
        - :class:`~asr_eval.models.gigaam_wrapper.GigaAMShortformCTC`
        - :class:`~asr_eval.models.gigaam_wrapper.GigaAMShortformRNNT`
    '''
    @abstractmethod
    def _get_model(self) -> GigaAMASR:
        """Instantiate a model."""
    
    def __init__(self):
        import gigaam
        from gigaam.model import SAMPLE_RATE, LONGFORM_THRESHOLD
        from gigaam.decoding import CTCGreedyDecoding
        
        if '_MODEL_HASHES' not in dir(gigaam):
            raise RuntimeError(
                'You seem to have installed old gigaam version from PyPI'
                ', update from https://github.com/salute-developers/GigaAM'
            )
        
        self.SAMPLE_RATE = SAMPLE_RATE
        self.LONGFORM_THRESHOLD = LONGFORM_THRESHOLD
        self.CTCGreedyDecoding = CTCGreedyDecoding

    @override
    def transcribe(self, waveform: FLOATS) -> str:
        import torch
        
        with torch.inference_mode():
            # a forward + decoding pipeline suitable for both CTC and RNNT
            model = self._get_model()
            inputs = (
                torch.tensor(waveform)
                .to(model._device)  # pyright:ignore[reportPrivateUsage]
                .to(model._dtype)  # pyright:ignore[reportPrivateUsage]
                .unsqueeze(0)
            )
            length = torch.full([1], inputs.shape[-1], device=model._device)  # pyright:ignore[reportPrivateUsage]
            encoded, encoded_len = model.forward(inputs, length)
            return model.decoding.decode(model.head, encoded, encoded_len)[0]
    

def _gigaam_load_model(
    model_name: str,
    fp16_encoder: bool = True,
    use_flash: bool | None = False,
    device: str | torch.device | None = None,
    download_root: str | None = None,
) -> GigaAM | GigaAMEmo | GigaAMASR:
    """Same as gigaam.load_model, but will remove half-downloaded
    checkpoints.
    """
    
    import gigaam
    try:
        return gigaam.load_model(
            model_name=model_name,
            fp16_encoder=fp16_encoder,
            use_flash=use_flash,
            device=device,
            download_root=download_root,
        )
    except RuntimeError as e:
        print(f'GigaAM failed to load: {e}')
        directory = (
            download_root
            if download_root is not None
            else gigaam._CACHE_DIR  # pyright: ignore[reportPrivateUsage]
        )
        print(f'Removing GigaAM cache dir {directory} and trying again')
        shutil.rmtree(directory)
        return gigaam.load_model(
            model_name=model_name,
            fp16_encoder=fp16_encoder,
            use_flash=use_flash,
            device=device,
            download_root=download_root,
        )


class GigaAMShortformRNNT(GigaAMShortformBase):
    """A GigaAM RNNT model. Supports different versions (see
    :code:`version` parameter): "v2", "v3", "v3_e2e".
    
    Installation: see :doc:`/guide_installation` page.
    """
    
    def __init__(
        self,
        version: Literal['v2', 'v3', 'v3_e2e'],
        device: str | torch.device = 'cuda',
        fp16: bool = False,
    ):
        from gigaam.model import GigaAMASR # type: ignore
        
        super().__init__()
        
        self._model = cast(GigaAMASR, _gigaam_load_model(
            f'{version}_rnnt', fp16_encoder=fp16, device=device
        ))
        self._model.eval()
    
    @override
    def _get_model(self) -> GigaAMASR:
        return self._model


class GigaAMShortformCTC(GigaAMShortformBase, CTC):
    """A GigaAM CTC model. Supports different versions (see
    :code:`version` parameter): "v2", "v3", "v3_e2e".
    
    Installation: see :doc:`/guide_installation` page.
    """
    
    def __init__(
        self,
        version: Literal['v2', 'v3', 'v3_e2e'],
        device: str | torch.device = 'cuda',
        fp16: bool = False,
    ):
        from gigaam.model import GigaAMASR # type: ignore
        from gigaam.decoding import Tokenizer
        from sentencepiece import SentencePieceProcessor
        
        super().__init__()
        
        self._model = cast(GigaAMASR, _gigaam_load_model(
            f'{version}_ctc', fp16_encoder=fp16, device=device
        ))
        self._model.eval()
        
        tokenizer = cast(Tokenizer, self._model.decoding.tokenizer)
        if tokenizer.charwise:
            # v2_ctc, v3_ctc have a vocab of characters
            vocab = self._model.decoding.tokenizer.vocab.copy()
            vocab.insert(self.blank_id, '')
        
        else:
            # v3_e2e_ctc has a vocab as a sentencepiece tokenizer
            sp = cast(
                SentencePieceProcessor,
                self._model.decoding.tokenizer.model
            )
            vocab_size = cast(int, sp.GetPieceSize())

            # check that the tokenizer does not contain byte fragments
            # so we can equivalently decode from vocab as `list[str]`
            assert [
                '\ufffd' not in sp.DecodeIds([id]) # type: ignore
                for id in range(sp.GetPieceSize()) # type: ignore
            ]

            vocab = [
                # replacing special char that means a space
                cast(str, sp.IdToPiece(id)).replace('▁', ' ') # type: ignore
                for id in range(vocab_size)
                # if not sp.IsControl(id) and not sp.IsUnknown(id)
            ]
            
        self._vocab = tuple(vocab)
    
    @override
    def _get_model(self) -> GigaAMASR:
        return self._model
    
    @override
    def transcribe(self, waveform: FLOATS) -> str:
        # we have two base classes: GigaAMShortformBase and CTC
        # usually we want to use GigaAMShortformBase base class to call
        # .transcribe()
        return super(GigaAMShortformBase, self).transcribe(waveform)
    
    @property
    @override
    def blank_id(self) -> int:
        return self._model.decoding.blank_id
    
    @property
    @override
    def tick_size(self) -> float:
        return 1 / FREQ
    
    @property
    @override
    def vocab(self) -> tuple[str, ...]:
        return self._vocab
    
    @override
    def ctc_log_probs(self, waveforms: list[FLOATS]) -> list[FLOATS]:
        import torch
        from torch.nn.utils.rnn import pad_sequence
        
        with torch.inference_mode():
            # Sampling rate should be equal to
            # gigaam.preprocess.SAMPLE_RATE == 16_000.
            assert isinstance(self._model.decoding, self.CTCGreedyDecoding)
            
            for waveform in waveforms:
                if len(waveform) / self.SAMPLE_RATE > self.LONGFORM_THRESHOLD:
                    warnings.warn(
                        'too long audio,'
                        ' GigaAMASR.transcribe() would throw an error',
                        RuntimeWarning
                    )
            
            waveform_tensors = [
                torch.tensor(w, dtype=self._model._dtype) # pyright: ignore[reportPrivateUsage]
                .to(self._model._device) # pyright: ignore[reportPrivateUsage]
                for w in waveforms
            ]
            lengths = (
                torch.tensor([len(w) for w in waveforms])
                .to(self._model._device) # pyright: ignore[reportPrivateUsage]
            )
            
            waveform_tensors_padded = pad_sequence(
                waveform_tensors,
                batch_first=True,
                padding_value=0,
            )
            
            encoded, encoded_len = self._model.forward(
                waveform_tensors_padded, lengths
            )
            
            log_probs = cast(
                torch.Tensor,
                self._model.head(encoder_output=encoded)
            )
            # exp(log_probs) sums to 1
            
            return [
                _log_probs[:length].cpu().numpy() # type: ignore
                for _log_probs, length in zip(log_probs, encoded_len)
            ]