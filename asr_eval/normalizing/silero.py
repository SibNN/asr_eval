from __future__ import annotations
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence, cast
import re
from string import printable, punctuation
import warnings

if TYPE_CHECKING:
    import torch

from asr_eval import CACHE_DIR
from asr_eval.utils.types import INTS


class RuSileroNormalizer:
    """ Converts numbers into words and makes other various
    normalization steps for evaluating WER on Russian text.
    
    A rare exception is handled that would create an inifinite loop,
    comparing with the original version. The normalizer is based on a
    neural network, so it is recommended to use caching.

    TODO release a model required for RuSileroNormalizer.
    
    Example:
        >>> from asr_eval.normalizing.silero import RuSileroNormalizer # doctest: +SKIP
        >>> from asr_eval.utils.cacheable import DiskCacheable # doctest: +SKIP
        >>> normalizer = RuSileroNormalizer() # doctest: +SKIP
        >>> normalizer = DiskCacheable(normalizer, cache_path='sliero_normalizer_cache.db') # doctest: +SKIP
        >>> print(normalizer('С 12.01.1943 г. площадь сельсовета — 1785,5 га.'))   # doctest: +SKIP
        С двенадцатого января тысяча девятьсот сорок третьего года
        площадь сельсовета — тысяча семьсот восемьдесят пять целых
        и пять десятых гектара
    
    The code is adapted from
    https://github.com/snakers4/russian_stt_text_normalization
    
    The model taken from (TODO upload the model to HF)
    https://t.me/silero_speech/6056
    """
    
    def __init__(
        self,
        model_path: str | Path = CACHE_DIR / 'silero_normalizer/jit_s2s.pt',
        device: str | torch.device | Literal['auto'] = 'auto',
    ):
        import torch
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.max_len = 150

        self.init_vocabs()

        print(f'Loading normalizer from {model_path}')
        self.model = cast(
            torch.jit.RecursiveScriptModule,
            torch.jit.load(model_path, map_location=device) # type: ignore
        )
        self.model.eval()
    
    def init_vocabs(self):
        # Initializes source and target vocabularies

        # vocabs
        rus_letters = (
            'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
            'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        )
        spec_symbols = '¼³№¾⅞½⅔⅓⅛⅜²'
        # numbers + eng + punctuation + space + rus
        self.src_vocab = {
            token: i + 5
            for i, token
            in enumerate(printable[:-5] + rus_letters + '«»—' + spec_symbols)
        }
        # punctuation + space + rus
        self.tgt_vocab = {
            token: i + 5
            for i, token in enumerate(punctuation + rus_letters + ' ' + '«»—')
        }

        unk = '#UNK#'
        pad = '#PAD#'
        sos = '#SOS#'
        eos = '#EOS#'
        tfo = '#TFO#'
        for i, token in enumerate([unk, pad, sos, eos, tfo]):
            self.src_vocab[token] = i
            self.tgt_vocab[token] = i

        self.unk_index = 0
        self.pad_index = 1
        self.sos_index = 2
        self.eos_index = 3
        self.tfo_index = 4

        inv_src_vocab = {v: k for k, v in self.src_vocab.items()}
        self.src2tgt = {
            src_i: self.tgt_vocab.get(src_symb, -1)
            for src_i, src_symb in inv_src_vocab.items()
        }

    def keep_unknown(self, string: str) -> tuple[str, list[str]]:
        reg = re.compile(r'[^{}]+'.format(''.join(self.src_vocab.keys())))
        unk_list = re.findall(reg, string)

        unk_ids = [
            range(m.start() + 1, m.end())
            for m in re.finditer(reg, string) if m.end() - m.start() > 1
        ]
        flat_unk_ids = [i for sublist in unk_ids for i in sublist]

        upd_string = ''.join([
            s for i, s in enumerate(string) if i not in flat_unk_ids
        ])
        return upd_string, unk_list

    def _norm_string(self, string: str) -> str:
        import torch
        
        # Normalizes chunk
        string_orig = string

        if len(string) == 0:
            return string
        string, unk_list = self.keep_unknown(string)

        token_src_list = [
            self.src_vocab.get(s, self.unk_index) for s in list(string)
        ]
        src = token_src_list + [self.eos_index] + [self.pad_index]

        src2tgt = [self.src2tgt[s] for s in src]
        

        src2tgt = torch.LongTensor(src2tgt).to(self.device)

        src = torch.LongTensor(src).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = cast(torch.Tensor, self.model(src, src2tgt))

        try:  # in asr_eval, a rare weird exception is handled here
            pred_words = self.decode_words(out, unk_list)
        except KeyError:
            warnings.warn(
                'Silero normalizer: KeyError occured in `decode_words`'
                ', returning the original string'
            )
            return string_orig.replace('⁇', ' ')


        if len(pred_words) > 199:
            warnings.warn("Sentence {} is too long".format(string), Warning)
        return pred_words

    def __call__(self, text: str) -> str:
        # Normalizes text

        # Splits sentences to small chunks with weighted length <= max_len:
        # * weighted length - estimated length of normalized sentence
        #
        # 1. Full text is splitted by "ending" symbols (\n\t?!.) to sentences;
        # 2. Long sentences additionally splitted to chunks: by spaces or just
        # dividing too long words
        
        # it is observed that the leading space (that Whisper usually predicts)
        # makes the model
        # work incorrectly, so we need to strip it
        text = text.strip()
        
        splitters = '\n\t?!'
        parts = [
            p
            for p in re.split(r'({})'.format('|\\'.join(splitters)), text)
            if p != ''
        ]
        norm_parts: list[str] = []
        for part in parts:
            if part in splitters:
                norm_parts.append(part)
            else:
                weighted_string = [7 if symb.isdigit() else 1 for symb in part]
                if sum(weighted_string) <= self.max_len:
                    norm_parts.append(self._norm_string(part))
                else:
                    spaces = [m.start() for m in re.finditer(' ', part)]
                    start_point = 0
                    end_point = 0
                    curr_point = 0

                    for i in count():
                        if start_point >= len(part):
                            break
                        if i > 1_000_000:
                            warnings.warn(
                                'RuSileroNormalizer looped for'
                                ' too long time, interrupting'
                            )
                            break
                        if curr_point in spaces:
                            if (
                                sum(weighted_string[start_point:curr_point])
                                < self.max_len
                            ):
                                end_point = curr_point + 1
                            else:
                                norm_parts.append(self._norm_string(
                                    part[start_point:end_point]
                                ))
                                start_point = end_point

                        elif (
                            sum(weighted_string[end_point:curr_point])
                            >= self.max_len
                        ):
                            if end_point > start_point:
                                norm_parts.append(self._norm_string(
                                    part[start_point:end_point]
                                ))
                                start_point = end_point
                            end_point = curr_point - 1
                            norm_parts.append(self._norm_string(
                                part[start_point:end_point]
                            ))
                            start_point = end_point
                        elif curr_point == len(part):
                            norm_parts.append(self._norm_string(
                                part[start_point:]
                            ))
                            start_point = len(part)

                        curr_point += 1
        
        return ''.join(norm_parts)

    def decode_words(
        self, pred: torch.Tensor, unk_list: Sequence[str] = ()
    ) -> str:
        pred_np = cast(INTS, pred.cpu().numpy()) # type: ignore
        return ''.join(self.lookup_words(
            x=pred_np,
            vocab={i: w for w, i in self.tgt_vocab.items()},
            unk_list=unk_list,
        ))

    def lookup_words(
        self, x: INTS, vocab: dict[int, str], unk_list: Sequence[str] = ()
    ) -> list[str]:
        unk_list = list(unk_list)
        result: list[str] = []
        for i in x:
            if i == self.unk_index:
                if len(unk_list) > 0:
                    result.append(unk_list.pop(0))
                else:
                    continue
            else:
                result.append(vocab[i])
        return [str(t) for t in result]
