Alignment features
###############################

Multi-reference alignment
********************************

The *asr_eval* supports aligning a prediction against multi-reference annotation. It also supports optional wildcard insertions that match any word sequence (possibly empty):

.. code-block:: none

    {Yeah|Yes|} it's {16|16-th|sixteenth}. <*> So, please {give me|gimme}

**Why?** Without being too time-consuming, such an annotation allows us to build more fair evaluation datasets for speech recognition. We see this as a more robust alternative to text normalization. Otherwise, during fine-tuning a model can adopt the dataset-specific annotation style, giving an illusion of metric improvement. The wildcard insertions allows us to skip annotating some phrases (that is, accept any model prediction as correct). This is especially important for annotating longform speech: you can't get rid of moments with poorly heard speech in a long enough audio recording.

So, this can be useful for:

- Labeling cluttered or disfluent speech
- Labeling longform speech with poorly heard moments
- For non-Latin languages or those with rich word formation

.. admonition:: Note

    While a naive approach exists to convert a multi-reference annotation into a list of single-variant annotations, and take a minimum WER accross them all, its complexity grows exponentially with the number of multi-reference blocks. This also would not support wildcard insertions. Our algorithm is quadratic in complexity but not exponential.

Enhanced alignment
********************************

In *asr_eval*, we produce higher quality alignments, both for streaming and non-streaming speech recognition. Across all the alignments optimal in WER (word error rate) sense, we search by several other criteria, such as character error rate and number of correct matches.

**Why?** In speech recognition, we usually align prediction and reference as follows: we find an alighment that is WER-optimal, that is, gives the least number of errors (replacements + insertions + deletions). However, there are many such alignments. Let the speaker say *"Multivariant... one, two, three"* which was transcribed as *"Multivariate"*. One of WER-optimal alignments aligns *"Multivariate"* with *"three"* (a replacement), and other words with nothing (deletions). This gives the false conclusion that the word *"three"* was already transribed. This will lead to incorrect calculation of the latency metric in a streaming speech recognition. Also, this would complicate a fine-grained WER analysis, where we want to analyze on which specific words the models made mistakes in.

So, this can be useful for:

- Better latency calculation for streaming recognition
- Better fine-grained error analysis

Multiple alignment
********************************

We provide an interactive tool to compare multiple predictions for the same annotated sample, with error highlighting, statistical testing and fine grained error comparison tool. It can be used to evaluate the quality of both models and datasets.

FAQ
********************************

**What languages ​​does asr_eval support?** It should work out of the box for many languages, however, some languages may require a custom word splitting method, a custom set of punctuation symbols to remove, a custom uppercase to lowercase mapping and other custom preprocessing like the replacement "ё" to "е" in Russian. These features are customizable.

**Does it replace text normalization?** Yes, and even more, it covers many cases where normalization doesn't help. It requires manual effort to annotate, but we believe that the test set should not be extremely large, so such an annotation is perfectly possible.

**How to annotate custom datasets?** We provide our experimental guidelines for annotators. However, we mainly focus on providing technical tools to work with multi-reference annotation; how to use it to annotate custom datasets is up for practioners.

**Does it complicate the annotation process?** We believe that multivariance usually makes things easier, because most cases where it is required are obvious to annotator, and you no more need to choose between equally acceptable options (and incorporate your own selection biases into a dataset). However, if you go further and try to make your annotation perfect, you find complex cases, where multi-reference annotation is controversial, but the percentage of such cases is small.

**Do you calculate CER (Character error rate)?** Not by default, but our alignment algorithm allows this, you just need to split the text into characters, not into words (see the example in the "High-level usage" section). It worth noting that WER-optimal word alignment and CER-optimal word alignment may be different - we focus on the first, and therefore do not calculate CER. In `AlignmentScore`, we calculate a total number of character errors in the WER-optimal alignment, but this is not the same as CER (see the `AlignmentScore` docs).

**Do you evaluate punctuation quality?** Currently no, we just remove punctuation before evaluating WER. However, our dataset `multivariant-v2` contains punctuation; it is slightly ambiguous how to put punctuation in multi-reference blocks.

Examples
********************************

This page describes how to use alignment features. For streaming features, see the "Streaming pipeline" page.


Preparing data
=========================

.. code-block:: python

    from asr_eval.align.parsing import DEFAULT_PARSER
    from asr_eval.align.alignment import Alignment, MultipleAlignment
    from asr_eval.align.solvers.dynprog import solve_optimal_alignment

    true = DEFAULT_PARSER.parse_multivariant_string(
        "{Yeah|Yes|} it's {16|16-th|sixteenth} . So, <*> please {give me|gimme}"
    )
    pred = DEFAULT_PARSER.parse_single_variant_string(
        "Yes it's sixteen. So, okay please give"
    )

Calculate sample WER
=========================

.. code-block:: python

    matches_list, _ = solve_optimal_alignment(true, pred)
    wer = matches_list.score.n_word_errors / matches_list.total_true_len
    print(f'{wer=:g}')

Output:

.. code-block:: none

    wer=0.222222

Display optimal alignment
============================

.. code-block:: python

    # .from_predictions internally calls `solve_optimal_alignment`
    alignment = Alignment.from_predictions(true, pred)
    msa = MultipleAlignment(true, alignments={'model': alignment})
    print(msa.view().render_as_text())

Output:

.. code-block:: none

    True  |  {Yeah|Yes|}  it  s  {16|16-th|sixteenth}  So  <*>   please  {give me|gimme}      
    model |  Yes          it  s  sixteen               So  okay  please  give                           

Or as HTML:

.. code-block:: python

    print(msa.view().render_as_text('html'))

Output:

.. raw:: html

    <span style="white-space: pre; font-family: 'Consolas', 'Ubuntu Mono', 'Monaco', monospace">True  |  {Yeah|Yes|}  it  s  {16|16-th|sixteenth}  So  <*>   please  {give me|gimme}      <br/>model |  Yes          it  s  <span style="background-color: #FF9C9C;">sixteen</span>               So  okay  please  give <span style="background-color: #FF9C9C;">               </span> </span>

Display multiple alignment
============================

.. code-block:: python

    pred2 = DEFAULT_PARSER.parse_single_variant_string(
        "yes it s six teen sir so please give me that"
    )
    pred3 = DEFAULT_PARSER.parse_single_variant_string(
        "Yes that's sixteenth, please give me that"
    )
    msa = MultipleAlignment(true, alignments={
        'model': Alignment.from_predictions(true, pred),
        'model2': Alignment.from_predictions(true, pred2),
        'model3': Alignment.from_predictions(true, pred3),
    })
    print(msa.view().render_as_text('html'))

Output:

.. raw:: html

    <span style="white-space: pre; font-family: 'Consolas', 'Ubuntu Mono', 'Monaco', monospace">True   |  {Yeah|Yes|}  it    s  {16|16-th|sixteenth}  So   <*>          please  {give me|gimme}          <br/>model  |  Yes          it    s  <span style="background-color: #FF9C9C;">sixteen</span>               So   okay         please  give <span style="background-color: #FF9C9C;">               </span>     <br/>model2 |  yes          it    s  <span style="background-color: #FF9C9C;">                    </span>  <span style="background-color: #FF9C9C;">six</span>  teen sir so  please  give me              <span style="background-color: #FF9C9C;">that</span><br/>model3 |  Yes          <span style="background-color: #FF9C9C;">that</span>  s  sixteenth             <span style="background-color: #FF9C9C;">  </span>                please  give me              <span style="background-color: #FF9C9C;">that</span></span>

Calculate CER
============================

The first option is to use a word pattern "." (any single character)

.. code-block:: python

    from asr_eval.align.parsing import Parser

    charwise_parser = Parser(tokenizing=r'.' )
    true = charwise_parser.parse_multivariant_string(
        "{Yeah|Yes|} it's {16|16-th|sixteenth} . So, <*> please {give me|gimme}"
    )
    pred = charwise_parser.parse_single_variant_string(
        "Yes it's sixteen. So, please give"
    )
    matches_list, _ = solve_optimal_alignment(true, pred)
    cer = matches_list.score.n_word_errors / matches_list.total_true_len
    print(f'{cer=:g}')

Output:

.. code-block:: none

    cer=0.219512

The second option using biopython wrapper (works only for single-variant annotation):

.. code-block:: python

    from asr_eval.align.char_aligner import char_align

    aligned = char_align(
        "Yeah it's sixteenth. So, please give me",
        "Yes it's sixteen. So, okay please give",
    )
    print(aligned)

    cer = (len(aligned.matching) - aligned.matching.count('|')) / len(aligned.matching)
    print(f'{cer=:g}')

Output:

.. code-block:: none

    CharAligned(
        first:    Yeah it's sixteenth. So, |||||please give me
        matching: ||.-|||||||||||||--||||||-----|||||||||||---
        second:   Yes| it's sixteen||. So, okay please give|||
    )
    cer=0.272727