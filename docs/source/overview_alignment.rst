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