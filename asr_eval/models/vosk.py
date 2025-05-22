from pathlib import Path
import math
import sys
from typing import Callable

import numpy as np
import numpy.typing as npt
import torch
from torch.nn.utils.rnn import pad_sequence
from huggingface_hub import hf_hub_download # type: ignore
import sentencepiece as spm # pyright: ignore[reportMissingTypeStubs]
import kaldifeat # pyright: ignore[reportMissingTypeStubs]
from icefall import NgramLm # pyright: ignore[reportMissingTypeStubs]
from icefall.lm_wrapper import LmScorer # pyright: ignore[reportMissingTypeStubs]
from icefall.utils import AttributeDict # pyright: ignore[reportMissingTypeStubs]


class VoskV54:
    def __init__(self, device: str | torch.device = 'cpu'):
        # adopted from https://huggingface.co/alphacep/vosk-model-ru/blob/main/decode.py
        
        revision = 'df6a54a4d8e5d43e82675e4f5dba2d507731a0d1'

        jit_script_path = hf_hub_download(
            'alphacep/vosk-model-ru',
            filename='am/jit_script.pt',
            revision=revision,
        )
        bpe_path = hf_hub_download(
            'alphacep/vosk-model-ru',
            filename='lang/bpe.model',
            revision=revision,
        )
        lm_path = hf_hub_download(
            'alphacep/vosk-model-ru',
            filename='lm/epoch-99.pt',
            revision=revision,
        )
        twogram_fst_path = hf_hub_download(
            'alphacep/vosk-model-ru',
            filename='lm/2gram.fst.txt',
            revision=revision,
        )
        code_path = hf_hub_download(
            'alphacep/vosk-model-ru',
            filename='decode.py',
            revision=revision,
        )
        
        self.code_dir = str(Path(code_path).parent)
        self.device = torch.device(device)
        
        self.model: torch.nn.Module = torch.jit.load(jit_script_path).to(self.device) # type: ignore
        self.model.eval() # type: ignore
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_path) # type: ignore
        
        opts = kaldifeat.FbankOptions() # type: ignore
        opts.device = self.device # type: ignore
        opts.frame_opts.dither = 3e-5 # type: ignore
        opts.frame_opts.snip_edges = False # type: ignore
        opts.frame_opts.samp_freq = 16000 # type: ignore
        opts.mel_opts.num_bins = 80 # type: ignore
        opts.mel_opts.high_freq = -400 # type: ignore
        self.fbank = kaldifeat.Fbank(opts) # type: ignore
        
        self.lm = LmScorer(
            lm_type="rnn",
            params=AttributeDict(
                lm_vocab_size=500,
                rnn_lm_embedding_dim=2048,
                rnn_lm_hidden_dim=2048,
                rnn_lm_num_layers=3,
                rnn_lm_tie_weights=True,
                lm_epoch=99,
                lm_exp_dir=str(Path(lm_path).parent),
                lm_avg=1,
            ),
            device=self.device,
            lm_scale=0.2,
        )
        self.lm.to(self.device)
        self.lm.eval()
        
        self.ngram_lm = NgramLm(
            twogram_fst_path,
            backoff_id=500,
            is_binary=False,
        )
        self.ngram_lm_scale = -0.1
        
        if self.code_dir not in sys.path:
            sys.path.append(self.code_dir)

        from decode import modified_beam_search_LODR # type: ignore
        self.modified_beam_search_LODR: Callable = modified_beam_search_LODR # type: ignore
        
    @torch.no_grad() # pyright: ignore[reportUntypedFunctionDecorator]
    def transcribe(self, waveforms: list[npt.NDArray[np.float64]]) -> list[str]:
        waveform_tensors = [torch.tensor(w, dtype=torch.float32).to(self.device) for w in waveforms]
        
        features = self.fbank(waveform_tensors)
        feature_lengths = [f.size(0) for f in features]
        
        features = pad_sequence(
            features,
            batch_first=True,
            padding_value=math.log(1e-10),
        )
        feature_lengths = torch.tensor(feature_lengths, device=self.device)
        
        encoder_out: torch.Tensor
        encoder_out_lens: torch.Tensor
        encoder_out, encoder_out_lens = self.model.encoder( # type: ignore
            features=features,
            feature_lengths=feature_lengths,
        )
        
        hyps: list[list[int]] = self.modified_beam_search_LODR( # type: ignore
            model=self.model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=20,
            LODR_lm=self.ngram_lm,
            LODR_lm_scale=self.ngram_lm_scale,
            LM=self.lm,
        )
        
        words: list[str] = [self.sp.decode(h) for h in hyps] # type: ignore
        
        return words