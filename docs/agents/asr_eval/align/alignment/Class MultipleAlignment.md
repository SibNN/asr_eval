# Class MultipleAlignment (defined in asr_eval/align/alignment.py at lines 566-883)

@dataclasses.dataclass
class MultipleAlignment:
    """A dataclass representing a group of multiple predicted texts
    aligned against the same ground truth text.

    Useful for displaying and aggregating purposes. May also store
    elapsed times if we want to compare several models by WER and
    inference time.

    For unlabeled data, all the predicted texts may be aligned against
    another predicted text (a shared baseline), in this case
    :attr:`~asr_eval.align.alignment.MultipleAlignment.baseline_name`
    contains its name, and the
    :class:`~asr_eval.align.alignment.MultipleAlignment` is considered
    unlabeled.

    Example:

        >>> from asr_eval.align.alignment import Alignment
        >>> from asr_eval.align.parsing import DEFAULT_PARSER
        >>> from asr_eval.align.alignment import MultipleAlignment
        >>> true = 'hey <*> {eh} one dollar'
        >>> preds = {
        ...     'first': 'Hey eh dollar',
        ...     'second': 'hey one dollar',
        ...     'third': 'Hey one dollar AB AB',
        ...     'fourth': 'Hey one dollar AB AB AB AB',
        ...     'fifth': '1 dollar!',
        ... }
        >>> true_parsed = DEFAULT_PARSER.parse_transcription(true)
        >>> alignments = {
        ...     name: Alignment(true_parsed,
        ...         DEFAULT_PARSER.parse_single_variant_transcription(pred))
        ...     for name, pred in preds.items()
        ... }
        >>> ma = MultipleAlignment(true_parsed, alignments)
        >>> print(ma.render_as_text(color_mode=None))
        true   |  hey  <*>               {eh}  one  dollar      
        first  |  Hey                    eh         dollar      
        second |  hey                          one  dollar      
        third  |  Hey                          one  dollar AB AB
        fourth |  Hey  one dollar AB AB        AB   AB          
        fifth  |  1                                 dollar      
        >>> ma.render_as_text(color_mode='html') # doctest: +ELLIPSIS
        '...'

    .. raw:: html

        <iframe src="_static/multiple_alignment_docstring.html"
            style="border: none; width: 100%; height: 130px; overflow: hidden;"></iframe>

    The examples show that the alignment is still not ideal for
    predictions 4 and 5. In the 4-th prediction, the visually best
    alignment would give 4 word errors, and the current alignment
    gives 2 errors, so, the goal to achieve a minimum number of word
    errors is to blame for this. In the 5-th prediction, the
    algorithm is not clever enough to known that "1" is closer to
    "one" than to "Hey". In this case, a multivariant annotation
    "{one|1}" or text normalization would help.
    """
    ...

    baseline: asr_eval.align.transcription.Transcription
    """The baseline against which all the predictions are aligned.
    Usually a ground truth.
    """

    alignments: dict[str, asr_eval.align.alignment.Alignment]
    """The alignments against the
    :attr:`~asr_eval.align.alignment.MultipleAlignment.baseline`.
    """

    elapsed_times: dict[str, float] = field(default_factory=dict[str, float])
    """The inference times for each prediction (may be NaN or absent).
    """

    baseline_name: str | typing.Literal['true'] = 'true'
    """The baseline name. Equals "true" when baseline is ground
    truth, otherwise an name of the
    :attr:`~asr_eval.align.alignment.MultipleAlignment.baseline`
    prediction against which other predictions are aligned.
    """

    def to_dataframe(self) -> pd.DataFrame:
        """Convert into a table view.

        For N alignments and M slots in the baseline (ground truth),
        returns NxM table. All the predicted words fill the table,
        in form of lists of
        :type:`~asr_eval.align.alignment.list[WORD_ERROR_TYPE]`. This is
        similar to Multiple Sequence Alignment (MSA) in biology and is
        used as an intermediate step in
        :meth:`~asr_eval.align.alignment.MultipleAlignment.render_as_text`.
        """
        ...

    def to_table(self) -> asr_eval.utils.table.Table2D[list[asr_eval.align.alignment.WORD_ERROR_TYPE]]:
        """Convert into a table view, but using a better typed
        :class:`~asr_eval.utils.table.Table2D` instead of
        :code:`pd.DataFrame`, compared to
        :meth:`~asr_eval.align.alignment.MultipleAlignment.to_dataframe`.
        """
        ...

    def render_as_text(
        self,
        color_mode: typing.Literal['ansi', 'html', None] = 'ansi',
        html_add_style: bool = True,
        add_prediction_names: bool = True,
        max_cell_size: int | None = 100,
        charwise_mode: bool = _CHARWISE_RENDER,
    ) -> str:
        """
        Visualizes all the alignments against the baseline (ground
        truth) as a multiline string.

        Args:
            color_mode: Colorize errors with ANSI color codes ("ansi")
                or html tags ("html").
            html_add_style: If True and :code:`color_mode="html"`, wraps
                the result in a <span> html tag with "white-space: pre"
                and monospace font. Such a font is important for visual
                alignment.
            add_prediction_names: If true, prepends lines with
                prediction names, as provided in the
                :attr:`~asr_eval.align.alignment.MultipleAlignment.alignments`.
            max_cell_size: Trims words larger than the specified size.
                May be useful for models that occasionally generate
                infinitely long words up to the generation limit.
            charwise_mode: Whether to disable separating words by space
                visually and mark deletions by a special character
                instead of space. Turn on for character-wise alignment.

        In the visualization, the outer slots of the baseline (i. e.
        words, gaps between them and multivariant blocks) are "columns",
        and predictions are rows.

        See example in the class docs.
        """
        ...