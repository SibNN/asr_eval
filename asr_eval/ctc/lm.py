from __future__ import annotations
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override, cast

import numpy as np

from asr_eval.segments.segment import TimedText

if TYPE_CHECKING:
    import kenlm
    from pyctcdecode.decoder import WordFrames

from asr_eval.models.base.interfaces import CTC, TimedTranscriber
from asr_eval.utils.types import FLOATS


__all__ = [
    'CTCDecoderWithLM',
    'OutputBeam',
]


@dataclass
class OutputBeam:
    """ Outputs of BeamSearchDecoderCTC.decode_beams as dataclass.
    
    Is needed for 0.5.0 version, but not for the last version from repo.
    In asr_eval we use the 0.5.0 version.
    """
    text: str
    last_lm_state: kenlm.State | list[kenlm.State] # type: ignore
    text_frames: list[WordFrames]
    logit_score: float
    
    # `lm_score` field is actially a combined LM + CTC score,
    # see https://github.com/kensho-technologies/pyctcdecode/issues/63
    lm_score: float


class CTCDecoderWithLM(TimedTranscriber):
    r""" Performs joint decoding from CTC logits and KenLM language
    model.
    
    Args:
        ctc_model: Any model with CTC interface.
        kenlm_path: A KenLM model path, usually a .gz or .bin file.
        unigrams: Passed into :code:`pyctcdecode.build_ctcdecoder`.
        alpha: Weight for language model during shallow fusion.
        beta: Weight for length score adjustment during scoring.
        beam_width: Passed into :code:`pyctcdecode.build_ctcdecoder`.
        unk_score_offset: Passed into
            :code:`pyctcdecode.build_ctcdecoder`.
        lm_score_boundary: Passed into
            :code:`pyctcdecode.build_ctcdecoder`.
        hotwords: A list of hotwords.
        hotword_weight: A score for hotwords.
        speedup_hotwords: If True, will try to speed up decoding with
            hotwords and make them not case sensitive, see below.
        beam_prune_logp: Passed into :code:`decoder._decode_logits`.
        token_min_logp: Passed into :code:`decoder._decode_logits`.
        
    Note:
        For Vosk models, it shows a warning: "No known unigrams
        provided, decoding results might be a lot worse." - this is ok
        (according to Nikolay V. Shmyrev)
        
        When using with CTCDecoderWithLM, it may show a warning "Found
        entries of length > 1 in alphabet. This is unusual unless style
        is BPE, but the alphabet was not recognized as BPE type. Is this
        correct?" - this is correct for wav2vec2, because
        :code:`.vocab()` in
        :class:`~asr_eval.models.wav2vec2_wrapper.Wav2vec2Wrapper` may
        contain special tokens like "<s>", but they usually are not
        predicted by the model (should have low logit scores).
    
    If :code:`speedup_hotwords=True`, will try to speed up decoding with
    hotwords. Otherwise uses a default pyctcdecode implementation that
    works as follows:
    
    .. code-block:: python
    
        # create pattern to match full words
        # sort by length to get longest possible match
        # use lookahead and lookbehind to match on word boundary instead of '\b' to only match
        # on space or bos/eos
        match_ptn = re.compile(
            r"|".join(
                [
                    r"(?<!\S)" + re.escape(s) + r"(?!\S)"
                    for s in sorted(hotword_unigrams, key=len, reverse=True)
                ]
            )
        )
        
        score = self._weight * len(self._match_ptn.findall(text))
    
    However, `hotword_unigrams` never contain space. To speedup, with
    :code:`speedup_hotwords=True` we replace it wil the following:
    
    .. code-block:: python
    
        hotwords: set[str]
        score = self._weight * sum([word in self.hotwords for word in text.split()])
    
    This should work equally. Note that when
    :code:`speedup_hotwords=True`, hotwords are not case sensitive,
    otherwise they are.
    """
    
    def __init__(
        self,
        ctc_model: CTC,
        kenlm_path: str | Path,
        unigrams: Collection[str] | None = None,
        alpha: float = 0.5,
        beta: float = 1.0,
        beam_width: int = 100,
        unk_score_offset: float = -10.0,
        lm_score_boundary: bool = True,
        hotwords: list[str] | None = None,
        hotword_weight: float = 10.0,
        speedup_hotwords: bool = True,
        beam_prune_logp: float = -10,
        token_min_logp: float = -5,
    ):
        import pyctcdecode
        
        # todo rewrite this check correctly
        # try:
        #     from pyctcdecode.decoder import OutputBeam # pyright: ignore[reportUnusedImport]
        # except ImportError:
        #     pass
        # else:
        #     raise RuntimeError(
        #         'You seem to have installed pyctcdecode from the repo directly.'
        #         ' Please install pyctcdecode from PyPI, example: pip install pyctcdecode==0.5.0'
        #         ' These versions are different, and we need to resort to the version from PyPI'
        #         ' to be compatible with other packages (such as T-One). Also note that numpy 2.x'
        #         ' is formally not supported for pyctcdecode==0.5.0, but it is actually compatible'
        #         ' - nothing will break.'
        #     )

        self.ctc_model = ctc_model
        self.beam_width = beam_width
        self.beam_prune_logp = beam_prune_logp
        self.token_min_logp = token_min_logp

        print('loading KenLM model...', flush=True)
        self.decoder = pyctcdecode.build_ctcdecoder(
            # pyctcdecode understands empty string as a blank token
            # (see pyctcdecode.alphabet._normalize_regular_alphabet)
            # this is the same convention as in the
            # asr_eval.models.base.interfaces.CTC.vocab()
            labels=list(self.ctc_model.vocab),
            kenlm_model_path=str(kenlm_path),
            unigrams=unigrams,
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            lm_score_boundary=lm_score_boundary,
        )
        self.hotwords = hotwords.copy() if hotwords else None
        self.hotword_weight = hotword_weight
        
        self.speedup_hotwords = speedup_hotwords
        if speedup_hotwords:
            from pyctcdecode.language_model import HotwordScorer
            
            class FasterHotwordScorer(HotwordScorer):
                def __init__(self, *args, **kwargs): # type: ignore
                    super().__init__(*args, **kwargs) # type: ignore
                    self.hotwords = cast(set[str], set(list(self._char_trie))) # type: ignore
                    self.hotwords = {w.lower() for w in self.hotwords}
                def score(self, text: str) -> float:
                    text = text.lower()
                    score = self._weight * sum(
                        word in self.hotwords for word in text.split() # type: ignore
                    )
                    return score
                # the HotwordScorer.score() method: to compare
                # def score(self, text: str) -> float:
                #     """Get total hotword score for input text."""
                #     return self._weight * len(self._match_ptn.findall(text))
            
            self.FasterHotwordScorerCls = FasterHotwordScorer
    
    def decode(self, waveform: FLOATS) -> list[OutputBeam]:
        """Accepts a waveform, returns beams, sorted from the best to
        the worst.
        """
        log_probs = self.ctc_model.ctc_log_probs([waveform])[0]
        return self.decode_from_log_probs(log_probs)
    
    def decode_from_log_probs(self, log_probs: FLOATS) -> list[OutputBeam]:
        """Accepts log probs from a CTC model, returns beams, sorted
        from the best to the worst.
        """
        from pyctcdecode.language_model import HotwordScorer
        from pyctcdecode.constants import DEFAULT_PRUNE_BEAMS
        # the default implementation `self.decoder.decode_beams` would work
        # slowly if hotwords are specified; we replicate `.decode_beams()`
        # below while changing HotwordScorer to the faster subclass
        
        # beams = self.decoder.decode_beams( # type: ignore
        #     log_probs,
        #     beam_width=self.beam_width,
        #     hotwords=self.hotwords,
        #     hotword_weight=self.hotword_weight,
        # )
        
        log_probs = log_probs.clip(np.log(1e-15), 0)
        
        if self.speedup_hotwords:
            hotword_scorer = self.FasterHotwordScorerCls.build_scorer(
                self.hotwords, weight=self.hotword_weight
            )
        else:
            hotword_scorer = HotwordScorer.build_scorer(
                self.hotwords, weight=self.hotword_weight
            )
        
        beams = self.decoder._decode_logits( # type: ignore
            log_probs,
            beam_width=self.beam_width,
            beam_prune_logp=self.beam_prune_logp,
            token_min_logp=self.token_min_logp,
            prune_history=DEFAULT_PRUNE_BEAMS,
            hotword_scorer=hotword_scorer,
            lm_start_state=None,
        )
        beams = [OutputBeam(*b) for b in beams] # type: ignore
        
        # sort from best to worst
        beams = sorted(beams, key=lambda b: -b.lm_score)
        
        return beams
        
    @override
    def timed_transcribe(self, waveform: FLOATS) -> list[TimedText]:
        top_beam = self.decode(waveform)[0]
        return [
            TimedText(
                start_tick * self.ctc_model.tick_size,
                end_tick * self.ctc_model.tick_size,
                text,
            )
            for text, (start_tick, end_tick) in top_beam.text_frames
        ]