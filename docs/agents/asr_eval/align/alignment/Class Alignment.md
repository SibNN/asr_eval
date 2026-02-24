# Class Alignment (defined in asr_eval/align/alignment.py at lines 248-508)

@dataclasses.dataclass(init=False)
class Alignment:
    """A word-to-word alignment between ground truth and prediction.

    The constructor internally runs
    :func:`~asr_eval.align.solvers.recursive.solve_optimal_alignment`,
    then places each of the predicted words into one of the
    :attr:`~asr_eval.align.alignment.Alignment.slots`. Each slot
    represents a specific position in the ground truth: before, at or
    after a word in the ground truth. See more examples and details
    in the user guide: :doc:`/guide_alignment_wer`.

    Example:
        >>> from asr_eval.align.alignment import Alignment
        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> true = 'Nothing hi there {one|1} {two|2} {eh} ok'
        >>> pred = 'No thing hi there one to eh oh'
        >>> true_parsed = DEFAULT_PARSER.parse_transcription(true)
        >>> pred_parsed = DEFAULT_PARSER.parse_single_variant_transcription(pred)
        >>> alignment = Alignment(true_parsed, pred_parsed)
        >>> # calculating WER
        >>> metrics, err_listing = alignment.error_listing()
        >>> metrics.word_error_rate()
        0.6666666666666666

        >>> # visualizing (run pip install jupyter to render HTML)
        >>> from IPython.display import HTML  # doctest: +SKIP
        >>> from asr_eval.align.alignment import MultipleAlignment
        >>> ma = MultipleAlignment(true_parsed, {'pred': alignment})
        >>> HTML(ma.render_as_text(color_mode='html'))  # doctest: +SKIP

        .. raw:: html

                <iframe src="_static/alignment_docstring.html"
                    style="border: none; width: 100%; height: 70px; overflow: hidden;"></iframe>

    Args:
        true: The first text to align, usually a ground truth. May be
            multivariant or single-variant, may include
            :class:`~asr_eval.align.transcription.Wildcard`
            insertions.
        pred: The second text to align, usually a prediction. Note that
            while the underlying alignment algorithm
            :func:`~asr_eval.align.solvers.recursive.solve_optimal_alignment`
            supports both texts to be multivariant, this class requires
            a single-variant prediction.
        absorb_insertions: If true, searches for insertions in "pre"
            slots (before or after ground truth words) and moves them into
            a neighbour "at" slot if this reduces CER. See the user guide
            :doc:`/guide_alignment_wer` for details.
    """
    ...

    true: asr_eval.align.transcription.TranscriptionPath
    """The parsed ground truth with selected path (i. e. a selected
    option index in each multivariant block, if such blocks exist).
    """

    pred: asr_eval.align.transcription.SingleVariantTranscription
    """The parsed prediction, as passed into the constructor."""

    slots: dict[asr_eval.align.transcription.OuterLoc | asr_eval.align.transcription.InnerLoc, list[asr_eval.align.alignment.WORD_ERROR_TYPE]]
    """The predicted words which are packed into slots. Each slot
    represents a specific position in the ground truth: before, at or
    after a word in the ground truth. See :doc:`/guide_alignment_wer`
    for details.
    """

    def error_listing(
        self,
        count_absorbed_insertions: bool = True,
        max_consecutive_insertions: int | None = None,
        skip_slots_with_zero_errors: bool = True,
    ) -> tuple[asr_eval.align.metrics.Metrics, dict[asr_eval.align.transcription.OuterLoc, asr_eval.align.alignment.ErrorListingElement]]:
        """Return WER metrics and detailed error positions.

        The first returned value is overall
        :class:`~asr_eval.align.metrics.Metrics` keeping a number
        of replacements, deletions and insertions. The WER value can
        be further obtained by 
        :meth:`~asr_eval.align.metrics.Metrics.word_error_rate`.

        The second returned value contains more detailed error analysis:
        a mapping from the ground truth outer slot (see the
        :doc:`/guide_alignment_wer` for details about slots) to
        :class:`~asr_eval.align.alignment.ErrorListingElement`.
        This may help in tasks such as error visualizing and
        fine-grained analysis.

        See examples in the
        :class:`~asr_eval.align.alignment.ErrorListingElement` docs.

        Args:
            max_consecutive_insertions: If set to integer N, in cases
                when more than N consecutive insertions occur between
                two words from the ground truth, count them as exactly N
                insertions. This helps to stabilize metric in the
                presence of ostillatory hallucinations, when the same
                word is repeated until the generation limit is reached.
                Also, this aligns better with common sense, where 100
                insertions in a row is not as big a problem as 100
                different errors in different places.
            count_absorbed_insertions: If the alignment was constructed
                with :code:`absorb_insertions=True`, should we count or
                skip the absorbed insertions? For example, in "nothing"
                vs "no thing", we will get a total of 2 errors with
                :code:`count_absorbed_insertions=True` and 1 error
                with :code:`False`. In the later case, our "WER"
                metric is different from the usual meaning, but is
                arguably better. However, the effect is often
                negligible.
            skip_slots_with_zero_errors: If True, will omit all the
            slots where there are zero word errors.
        """
        ...

    def render_as_text(
        self,
        color_mode: typing.Literal['ansi', 'html', None] = 'ansi',
        html_add_style: bool = True,
        max_cell_size: int | None = 100,
        name: str | None = 'pred',
    ) -> str:
        """Visualizes the alignment. Returns a string contatining
        of lines: the ground truth and the prediction.

        :code:`name` argument specifies how the prediction will
        be titled in the output string. If None, will not add name.

        For other arguments, see
        :meth:`~asr_eval.align.alignment.MultipleAlignment.render_as_text`.
        """
        ...

    def get_true_len(self) -> int:
        """Get the words count in the ground truth.

        If there are multivariant blocks in the ground truth, selects
        the shortest (possibly empty) option in each block.

        Also, :class:`~asr_eval.align.transcription.Wildcard` tokens do
        not increment the ground truth length.
        """
        ...