Streaming pipeline
##############################

On this page we will go through all stages of the streaming evaluation pipeline, from preparing time labeled data to diagramming.

The code below is from the `notebooks/Demo.ipynb` notebook.

.. raw:: html

   <details>
   <summary><a>The required imports</a></summary>

.. code-block:: python

   def fn():

.. raw:: html

   </details>

Multivariant format
**************************************

In `asr_eval`, a transcription may contain zero or more of the following elements:

1. **Multivariant blocks** in braces, where options are separated by vertical line "|". If there is only a single option in braces, empty substring is added as the other option: *{oh}* equals *{oh|}*, but *{oh|uh}* does not equal *{oh|uh|}*. Currently, no nested braces are allowed.

2. **"Anything" blocks**, typed as `<*>`, that match any sequence of words, possibly empty. These blocks however do not increase the counter of the correctly transcribed words. They represent our refusal to evaluate a certain segment.

An example: *{Okay|okey|ok} {Okay|okey|ok|} I give {one dollar|1 dollar|1$|$1}*.

Here a person speaks "okay" twice, but some models such as Whisper may omit the second word, as people transcribers would do, and we may not want to consider it a mistake.

Tokenizing transcriptions
*******************************

When annotating a dataset, we should be aware of the tokenization used. For example, if "1$" is tokenized as a single word, then we should extend our multivariant block to contain both "1$" and  "1 $" (with space). Based on this, we can formulate requirements for tokenization rules:

1. Make it simpler so as not to confuse annotators.
2. Tokenize in a way that reduces number of variants in multivariant blocks.
3. Make it complementary with the subsequent stage of punctuation removal.

A well-known :code:`nltk.wordpunct_tokenize` algorithm searches for either :code:`\\w+` or :code:`[^\\w\\s]+` matches. This does not allow to separate symbols such as dollar sign from punctuation: *"$!"* will be considered a single token.

**Our algorithm**

We treat the following symbols as a punctuation :code:`{P} = .,!?:;…-–—'"‘“”«»()[]{}` and search one of the following patterns:

1. :code:`\\w+` (a sequence of word symbols: letters and digits)
2. :code:`[^\\w\\s{P}]+` (a sequence of other symbols, excluding spaces)
3. :code:`[{P}]+` (a sequence of punctuation symbols)

The first two patterns are considered words.

Our :code:`parse_multivariant_string` function supports 4 methods: ours ("asr_eval"), "wordpunct_tokenize", "space" (searching for :code:`\S+`) and "`razdel`_" (a library for Russian word tokenization). Based on the arguments above, we set "asr_eval" as a default method. The function also applies postprocessing (enabled by default, can be disabled):

.. _razdel: https://github.com/natasha/razdel

1. Drops punctiation words (`[{P}]+`).
2. Turns into lower case.
3. Converts Russian "ё" into "е".

We keep the positions of each word in the original text. In the example below, words are stored as lower case, but spans are visualized in the original text and thus may be in upper case. Token colors have no meaning and are just to visually separate them.

.. code-block:: python

    text = (
        '(7-8 мая) в Пуэрто-Рико прошёл {шестнадцатый|16-й|16}'
        ' этап "Формулы-1" с фондом 100,000$!'
    )

    for method in 'space', 'razdel', 'wordpunct_tokenize', 'asr_eval':
        tokens = parse_multivariant_string(text, method=method)
        colored_str, colors = colorize_parsed_string(text, tokens)
        print(f'{method: <20}', colored_str)

.. image:: images/tokenization.png

Obtaining timed transcriptions
************************************

We use CTC force alignment to determine time span for each symbol. While CTC loss does not enforce correct positioning, we notice that such a pseuco-labeling is precise enough: an error is usually less than 0.2 seconds and rarely exceeds 0.5 seconds.

For multivariant blocks, we need at least one option that can be encoded into model's token ID. For example, Russian GigaAM 2 model has a vocabulary of russian letters only, and in a block "{Facebook|Фейсбук}" only the last option can be encoded. If several options can be encoded, we select the longest one, for example in "{milliseconds|ms}" we select the first option. After selecting one option in each block, we can perform force alignment. If this is not possible (very rare cases), and we throw an exception.

After obtaining timings for one option for a multivariant block, we can propagate timings into the other options. Each option is a list of words. Let we have two options :code:`A = X + A1 + Y` (timed) and :code:`B = X + B1 + Y`, where some lists may be empty. Let either :code:`len(A) == 1 and len(B) >= 1` or :code:`len(A) >= 1 and len(B) == 1`. In these cases we can assign timings for :code:`B` given timings for :code:`A`.

The function `fill_word_timings_inplace` accepts CTC model, waveform, a tokenized transcription (possibly multivariant) and performs force alignments with filling :code:`.start_time` and :code:`.end_time` for each word. Set :code:`verbose=True` to see the process of timing propagation.

.. code-block:: python

    waveform: npt.NDArray[np.floating] = (
        librosa.load('tests/testdata/formula1.mp3', sr=16000)[0])
    waveform += waveform[::-1] / 4  # add some speech-like noise

    text = Path('tests/testdata/formula1.txt').read_text()
    tokens = parse_multivariant_string(text)

    model = typing.cast(GigaAMASR, gigaam.load_model('ctc', device='cuda'))
    fill_word_timings_inplace(model, waveform, tokens, verbose=True)

Output:

.. code-block::

    Propagated timings from [седьмого (0.5-1.0)] to [7 (0.5-1.0)]
    Propagated timings from [восьмого (1.2-1.7)] to [8 (1.2-1.7)]
    Propagated timings from [шестнадцатый (4.9-5.8)] to [16й (4.9-5.8)]
    Propagated timings from [шестнадцатый (4.9-5.8)] to [16 (4.9-5.8)]
    Propagated timings from [шестнадцатый (4.9-5.8)] to [16 (4.9-5.4), й (5.4-5.8)]
    Propagated timings from [формулы (6.6-7.1), один (7.3-7.7)] to [формулы (6.6-7.1), 1 (7.3-7.7)]
    Propagated timings from [сто (9.0-9.3), тысяч (9.5-9.8), долларов (10.0-10.5)] to [100 (9.0-9.3), тысяч (9.5-9.8), долларов (10.0-10.5)]
    Propagated timings from [100 (9.0-9.3), тысяч (9.5-9.8), долларов (10.0-10.5)] to [100 (9.0-9.3), тыщ (9.5-9.8), долларов (10.0-10.5)]
    Propagated timings from [100 (9.0-9.3), тысяч (9.5-9.8), долларов (10.0-10.5)] to [100 (9.0-9.3), 000 (9.5-9.8), долларов (10.0-10.5)]
    ...

We can visualize the result and the waveform:

.. code-block:: python

    plt.figure(figsize=(15, 4))
    plt.plot(np.arange(len(waveform)) / 16000,
        3 * waveform / waveform.max(), alpha=0.3, zorder=-1)
    draw_timed_transcription(tokens, y_delta=-3)
    plt.ylim(-3.5, 3.5)
    plt.show()

    print(colorize_parsed_string(text, tokens)[0])

.. image:: images/multivariant_waveform.png

Note that having 14 options in a multivariant block is a very rare situation, most cases are much simpler.

Streaming model
**************************

We provide a detailed docstring in the `StreamingASR` class. The main features are described in the overview section :ref:`Preparing streaming models`. For know it is important that a streaming model accepts waveform chunks and returns transcription chunks (adding more words or editing previous words), and they are not always related one to one.

Streaming evaluation
**************************

A :code:`default_evaluation_pipeline` function starts sending input chunks, receives the full transcription and evaluates it against the ground truth. To customize, you can copy and edit the function contents.

.. code-block:: python

    asr = VoskStreaming(model_name='vosk-model-ru-0.42', chunk_length_sec=1)
    asr.start_thread()

    recording = Recording(
        transcription=text,
        transcription_words=tokens,
        waveform=waveform,
    )
    eval = default_evaluation_pipeline(recording, asr)

    asr.stop_thread()

The result :code:`eval.partial_alignments: list[PartialAlignment]` is the main concept in the streaming evaluation pipeline. Each partial alignment is a state at a certain point in time. It keeps 3 time points:

- :code:`at_time`. A real time of interest.
- :code:`audio_seconds_sent`. Audio seconds sent into the model: end time of the last input chunk sent before `at_time`.
- :code:`audio_seconds_processed`. Audio seconds processed. The model returns this value with each output chunk, and we take the value from the last output chunk received before `at_time`.

For each partial alignment, the prediction is a union of all output chunks received before `at_time`, and the field `.pred` contains a tokenized version of the transcription.

.. code-block:: python

    print(TranscriptionChunk.join(eval.output_chunks))
    print(eval.partial_alignments[-1].pred)

Output:

.. code-block::

    седьмого восьмого мая по эру дарика прошёл шестнадцатый этаж формулы один с фондом сто тысяч долларов победителем стал гонщик мерседеса
    [Token(седьмого), Token(восьмого), Token(мая), Token(по), Token(эру), Token(дарика), Token(прошел),
    Token(шестнадцатый), Token(этаж), Token(формулы), Token(один), Token(с), Token(фондом), Token(сто),
    Token(тысяч), Token(долларов), Token(победителем), Token(стал), Token(гонщик), Token(мерседеса)]

For each partial alignment, we take the beginning of the true transcription until :code:`audio_seconds_processed` and align it with the prediction. This works also for multivariant transcriptions. If :code:`audio_seconds_processed` is in the middle of a word, we consider two options with and without this word, and select one with the lowest word error count. :code:`PartialAlignment.alignment.matches` contains a list of :code:`Match`, where each match has one of the following statuses: "correct", "deletion", "insertion", or "replacement".

.. code-block:: python

    eval.partial_alignments[-1].alignment.matches

Output:

.. code-block::

    [Match(Token(седьмого, t=(0.5, 1.0)), Token(седьмого)),
    Match(Token(восьмого, t=(1.2, 1.7)), Token(восьмого)),
    Match(Token(мая, t=(1.9, 2.2)), Token(мая)),
    Match(Token(в, t=(2.5, 2.6)), Token(по)),
    Match(Token(пуэрто, t=(2.7, 3.3)), Token(эру)),
    Match(Token(рико, t=(3.5, 3.8)), Token(дарика)),
    ...

Each :code:`Match` can be converted into a :code:`StreamingASRErrorPosition`. It is similar to match, but:

1. Always keeps an audio timings to display. In :code:`Match`, for "insertion" we have no timings, which is obvious. When converting to :code:`StreamingASRErrorPosition`, we assign a timing between neighbour words, just to be able to visualize.

2. Can have a status "not_yet": which is assigned if a Match is a "deletion", and all subsequent matches until :code:`audio_seconds_processed` are "deletion". In this way we can distinguish between missing and not yet transcribed words.

.. code-block:: python

    eval.partial_alignments[-1].get_error_positions()

Output:

.. code-block::

    [StreamingASRErrorPosition(start_time=0.48, end_time=1.0, processed_time=16.031375, status='correct'),
    StreamingASRErrorPosition(start_time=1.2, end_time=1.72, processed_time=16.031375, status='correct'),
    StreamingASRErrorPosition(start_time=1.92, end_time=2.2, processed_time=16.031375, status='correct'),
    StreamingASRErrorPosition(start_time=2.52, end_time=2.64, processed_time=16.031375, status='replacement'),
    ...
    StreamingASRErrorPosition(start_time=13.48, end_time=14.12, processed_time=16.031375, status='correct'),
    StreamingASRErrorPosition(start_time=14.36, end_time=14.84, processed_time=16.031375, status='not_yet'),
    StreamingASRErrorPosition(start_time=15.08, end_time=15.6, processed_time=16.031375, status='not_yet')]

We can visualize all the error positions, sent and processed times on a diagram.

.. code-block:: python

    plt.figure(figsize=(15, 6))
    partial_alignments_plot(eval)
    plt.show()

.. image:: images/partial_alignment_plot.png

From the diagram we can make the following observations:

1. The processed time (dark green line) lags up to one second behind the sent time (gray line). This is because the model uses a rechunking with accumulation of one second of audio.

2. The model is able to successfully correct some words as more audio data arrives, but there is one word with with the opposite situation.

3. The model failed to recognize the last words.