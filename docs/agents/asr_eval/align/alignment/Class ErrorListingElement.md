# Class ErrorListingElement (defined in asr_eval/align/alignment.py at lines 85-210)

@dataclasses.dataclass
class ErrorListingElement:
    """Info about errors for a specific slot in the ground truth
    transcription.

    Note:
        This class enables fine-grained error analysis and is not needed
        if you only want to calculate WER.

    See more info about slots in the user guide:
    :doc:`/guide_alignment_wer`. In short, each slot represents a
    specific position in the ground truth: before, at or after a word in
    the ground truth. :code:`ErrorListingElement` keeps a list of
    insertions, deletions or replacements for a specific slots. Thus, it
    represents errors made in the specific place of the ground truth
    transcription. Note that there may be multiple predicted words for a
    single slot (for example, a multi-word insertion), and thus multiple
    errors. Therefore, :code:`.pred` field contains a list of errors.

    The :code:`ErrorListingElement` may additionally keep a
    :code:`.sample_id`. We can gather error listings from many samples
    and merge them into a joint :code:`list[ErrorListingElement]`. This
    list represents a full error statistics on a dataset. We can group
    the list by :code:`.true_text` field and obtain error statistics for
    different words. This is what happening in
    :func:`~asr_eval.bench.evaluator.compare_pipelines` that is used by
    a dashboard.

    Example:
        >>> from collections import defaultdict
        >>> from asr_eval.align.alignment import Alignment, ErrorListingElement
        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> from asr_eval.align.metrics import Metrics
        ... 
        >>> truth_and_predictions = [
        ...     ('Alexa, turn the light on.', 'alex turns the light on'),
        ...     ('Alexa, scenario off.', 'alex scene area off'),
        ...     ('Alexa, play music.', 'alexa play music'),
        ...     ('Alexa, turn if off, thanks.', 'alex turns if off thanks'),
        ...     ('Alexa, hello!', 'alexa hello'),
        ... ]
        ... 
        >>> metrics = Metrics()
        >>> errors: dict[str, list[ErrorListingElement]] = defaultdict(list)
        >>> for true, pred in truth_and_predictions:
        ...     alignment = Alignment(
        ...         DEFAULT_PARSER.parse_single_variant_transcription(true),
        ...         DEFAULT_PARSER.parse_single_variant_transcription(pred),
        ...     )
        ...     metric, listing = alignment.error_listing(
        ...         skip_slots_with_zero_errors=False)
        ...     metrics += metric  # uses Metrics.__add__ overloading
        ...     for listing_element in listing.values():
        ...         if listing_element.true_text is not None:
        ...             errors[listing_element.true_text].append(listing_element)
        >>> print(metrics)
        Metrics(true_len=18, n_replacements=6, n_insertions=1, n_deletions=0)

        >>> for word, error_statistics in errors.items():
        ...     good_cases = [x for x in error_statistics if x.n_errors == 0]
        ...     bad_cases = [x for x in error_statistics if x.n_errors > 0]
        ...     if len(bad_cases):
        ...         print(
        ...             f'{word}: {len(good_cases)} correct, {len(bad_cases)}'
        ...             f' wrong: {[x.pred_text for x in bad_cases]}'
        ...         )
        alexa: 2 correct, 3 wrong: ['alex', 'alex', 'alex']
        turn: 0 correct, 2 wrong: ['turns', 'turns']
        scenario: 0 correct, 1 wrong: ['scene']
    """
    ...

    outer_loc: asr_eval.align.transcription.OuterLoc
    """An outer slot in the ground truth transcription. See more info
    about slots in the user guide: :doc:`/guide_alignment_wer`.
    """

    true: asr_eval.align.transcription.Token | asr_eval.align.transcription.MultiVariantBlock | None
    """For an "at" slot contains the correxponding block. For a "pre"
    slot is None.
    """

    true_text: str | None
    """A joint ground truth text for the given slot. For a "pre" slot is
    None.
    """

    pred: list[asr_eval.align.alignment.WORD_ERROR_TYPE]
    """A list of everything that was predicted for the current slot. May
    contain correct matches, insertions, deletions or replacements.
    """

    n_replacements: int
    """Number of :class:`~asr_eval.align.alignment.Replacement` elements
    in the :code:`.pred`."""

    n_insertions: int
    """Number of :class:`~asr_eval.align.alignment.Insertion` elements
    in the :code:`.pred`, which may be clipped above by
    :code:`max_consecutive_insertions` parameter or corrected by
    :code:`count_absorbed_insertions` parameter of
    :meth:`~asr_eval.align.alignment.Alignment.error_listing`.
    """

    n_deletions: int
    """Number of :class:`~asr_eval.align.alignment.Deletion` elements
    in the :code:`.pred`.
    """

    sample_id: int | None = None
    """A sample id in the dataset, is None by default and can be filled
    manually.
    """

    @property
    def n_errors(self) -> int:
        """A sum of n_replacements, n_insertions and n_deletions."""
        ...

    @property
    def pred_text(self) -> str:
        """A joint predicted text obtained from :code:`.pred` field."""
        ...