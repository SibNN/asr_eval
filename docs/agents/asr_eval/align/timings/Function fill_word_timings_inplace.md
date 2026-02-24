# Function fill_word_timings_inplace (defined in asr_eval/align/timings.py at lines 197-407)

def fill_word_timings_inplace(
    model: asr_eval.models.base.interfaces.CTC,
    waveform: asr_eval.utils.types.FLOATS,
    transcription: asr_eval.align.transcription.Transcription,
    verbose: bool = False,
):
    """Fills :attr:`~asr_eval.align.transcription.Token.start_time` and
    :attr:`~asr_eval.align.transcription.Token.end_time` in
    transcription via forced alignment.

    Args:
        model: A model with
            :class:`~asr_eval.models.base.interfaces.CTC` interface.
            Normally it should support all the characters found in
            `transcription` ignoring case. However, if it does not
            support some options in a multivariant block, the function
            will run some propagation rules and try to fill the timings.
            For example, if there is no digits in the model's vocab,
            it will still able to fill the timings for a block
            :code:`{1|one}`. We first fill the timings for the word
            "one", then mirror them to the word "1".
        waveform: The audio 16000 kHz, possibly long. For long audios
            will wrap the CTC model into a
            :class:`~asr_eval.models.base.longform.LongformCTC` that
            works via logit averaging of uniform chunks with overlap.
        transcription: A single-variant or multivariant transcription.
        verbose: Enable debug output.

    Raises:
        CannotFillTimings: if cannot fill timings due to the absence of
            the required characters in the model's vocab, or other
            limitations of the algorithm.

    Example:
        >>> from typing import cast
        >>> from datasets import load_dataset, Audio
        >>> from asr_eval.align.timings import fill_word_timings_inplace
        >>> from asr_eval.bench.datasets import get_dataset
        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> from asr_eval.align.plots import draw_timed_transcription
        >>> dataset = (
        ...     load_dataset('PolyAI/minds14', name='en-GB', split='train')
        ...     .cast_column('audio', Audio(16_000))
        ... )
        >>> sample = dataset[0]
        >>> transcription = DEFAULT_PARSER \\
        ...     .parse_transcription(sample['transcription'])
        >>> waveform = sample['audio']['array']

        >>> # # to display the audio:
        >>> # import IPython.display    # doctest: +SKIP
        >>> # IPython.display.Audio(waveform, rate=16_000)  # doctest: +SKIP

        >>> # For English:
        >>> from asr_eval.models.wav2vec2_wrapper import Wav2vec2Wrapper
        >>> model = Wav2vec2Wrapper('facebook/wav2vec2-base-960h')

        >>> # # For Russian:
        >>> # from asr_eval.models.gigaam_wrapper import GigaAMShortformCTC
        >>> # model = GigaAMShortformCTC('v2')

        >>> fill_word_timings_inplace(model, waveform, transcription)
        >>> print(transcription.blocks[:6]) # doctest: +NORMALIZE_WHITESPACE
        (Token(i, t=(1.0, 1.0)), Token(want, t=(1.1, 1.3)), Token(to, t=(1.3, 1.5)),
         Token(pay, t=(2.0, 2.2)), Token(a, t=(2.3, 2.4)), Token(bill, t=(2.4, 2.7)))

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> plt.figure(figsize=(8, 2)) # doctest: +ELLIPSIS
        <...>
        >>> draw_timed_transcription(transcription, y_tick_width=0.02) # doctest: +ELLIPSIS
        >>> plt.plot(np.arange(len(waveform)) / 16000, waveform, alpha=0.3) # doctest: +ELLIPSIS
        [...]

    .. image:: images/docstrings/fill_word_timings_inplace.png

    """
    ...