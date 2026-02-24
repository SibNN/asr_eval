# Class VoxtralWrapper (defined in asr_eval/models/voxtral_wrapper.py at lines 16-89)

class VoxtralWrapper(asr_eval.models.base.openai_wrapper.APITranscriber):
    """A wrapper to call Voxtral via OpenAI API.

    Installation: see :doc:`/guide_installation` page.

    Example:
        >>> voxtral = VoxtralWrapper('mistralai/Voxtral-Mini-3B-2507') #doctest: +SKIP
        >>> text = voxtral.transcribe(speech_sample(repeats=2)) #doctest: +SKIP
        >>> print(text) #doctest: +SKIP
        >>> voxtral.stop_vllm_server() #doctest: +SKIP

    See the VLLM source code in
    :code:`vllm.model_executor.models.voxtral`.

    According to :code:`VoxtralEncoderModel.prepare_inputs_for_conv`,
    the Voxtral pipeline splits a long audio into non-overlapping
    chunks, then processes each chunk via Whisper and concatenate the
    outputs. So, the LLM sees the whole long audio at once.

    According to
    :code:`vllm.model_executor.models.voxtral.get_generation_prompt`,
    the Voxtral uses :code:`encode_transcription` method of
    :code:`mistral_common.tokens.tokenizers.instruct.InstructTokenizerV7`
    tokenizer. It starts from <bos>, adds audio, adds
    f"lang:{request.language}" substring and a special token
    [TRANSCRIBE].

    Thus, there is a problem with using domain words in Voxtral, since
    such a prompt does not support user instructions. There may be
    solutions, but this feature is not implemented in this wrapper yet.

    Authors: Vasily Kudryavtsev & Oleg Sedukhin
    """
    ...

    @typing.override
    def vllm_run_args(self) -> list[str]:
    ...