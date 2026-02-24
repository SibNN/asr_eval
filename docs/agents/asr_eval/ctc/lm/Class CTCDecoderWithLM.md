# Class CTCDecoderWithLM (defined in asr_eval/ctc/lm.py at lines 42-251)

class CTCDecoderWithLM(asr_eval.models.base.interfaces.TimedTranscriber):
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
    ...

    def decode(self, waveform: asr_eval.utils.types.FLOATS) -> list[asr_eval.ctc.lm.OutputBeam]:
        """Accepts a waveform, returns beams, sorted from the best to
        the worst.
        """
        ...

    def decode_from_log_probs(self, log_probs: asr_eval.utils.types.FLOATS) -> list[asr_eval.ctc.lm.OutputBeam]:
        """Accepts log probs from a CTC model, returns beams, sorted
        from the best to the worst.
        """
        ...

    @typing.override
    def timed_transcribe(self, waveform: asr_eval.utils.types.FLOATS) -> list[asr_eval.segments.segment.TimedText]:
    ...