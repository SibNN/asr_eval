from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import os
from typing import Literal, cast

import numpy as np
import pandas as pd
from termcolor import colored

from asr_eval.align.solvers.dynprog import solve_optimal_alignment
from asr_eval.align.metrics import Metrics
from asr_eval.utils.table import Table2D
from asr_eval.align.matching import (
    Match, _get_len, char_edit_distance, _select_shortest_multi_variants# pyright: ignore[reportPrivateUsage]
)
from asr_eval.align.transcription import (
    MultiVariantBlock,
    Transcription,
    TranscriptionPath,
    SingleVariantTranscription,
    Token,
    OuterLoc,
    InnerLoc,
    get_outer_slots,
    get_outer_slots_values,
)


__all__ = [
    'Alignment',
    'MultipleAlignment',
    'ErrorListingElement',
    'WORD_ERROR_TYPE',
    'Deletion',
    'Correct',
    'Replacement',
    'Insertion',
]


@dataclass
class _TokenContainerMixin:
    token: Token
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.token!r})'


class Deletion:
    """One of the error types in
    :meth:`~asr_eval.align.alignment.Alignment.error_listing`
    """
    def __repr__(self) -> str:
        return 'Deletion()'


class Correct(_TokenContainerMixin):
    """One of the error types in
    :meth:`~asr_eval.align.alignment.Alignment.error_listing`
    """
    pass


class Replacement(_TokenContainerMixin):
    """One of the error types in
    :meth:`~asr_eval.align.alignment.Alignment.error_listing`
    """
    pass


class Insertion(_TokenContainerMixin):
    """One of the error types in
    :meth:`~asr_eval.align.alignment.Alignment.error_listing`
    """
    pass


WORD_ERROR_TYPE = Correct | Replacement | Insertion | Deletion
"""A wrapper to store a :class:`~asr_eval.align.transcription.Token`
and an associated error type.
"""


@dataclass
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
    
    outer_loc: OuterLoc
    """An outer slot in the ground truth transcription. See more info
    about slots in the user guide: :doc:`/guide_alignment_wer`.
    """
    
    true: Token | MultiVariantBlock | None
    """For an "at" slot contains the correxponding block. For a "pre"
    slot is None.
    """
    
    true_text: str | None
    """A joint ground truth text for the given slot. For a "pre" slot is
    None.
    """
    
    pred: list[WORD_ERROR_TYPE]
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
        return self.n_replacements + self.n_insertions + self.n_deletions
    
    @property
    def pred_text(self) -> str:
        """A joint predicted text obtained from :code:`.pred` field."""
        return ' '.join([
            ('' if isinstance(x, Deletion) else x.token.to_text())
            for x in self.pred
        ])


def _matches_to_slots(
    true: TranscriptionPath, matches: list[Match]
) -> dict[OuterLoc | InnerLoc, list[WORD_ERROR_TYPE]]:
    slots: dict[OuterLoc | InnerLoc, list[WORD_ERROR_TYPE]] = defaultdict(list)
    
    last_true_slot_idx: int | None = None
    for match in matches:
        if match.true is not None:
            slot_idx, slot_loc = true.token_uid_to_slot(match.true.uid)
            last_true_slot_idx = slot_idx
            if match.pred is not None:
                if match.status == 'correct':
                    # correct
                    slots[slot_loc].append(Correct(match.pred))
                else:
                    # replacement
                    assert match.status == 'replacement'
                    slots[slot_loc].append(Replacement(match.pred))
            else:
                if match.status != 'correct':
                    # may be correct for unmatched Wildcard()
                    # deletion
                    slots[slot_loc].append(Deletion())
        else:
            # insertion
            if last_true_slot_idx is None:
                # before the first true token
                slot_loc = true.slot_idx_to_loc(0)
            else:
                slot_loc = true.slot_idx_to_loc(last_true_slot_idx + 1)
            assert match.pred is not None
            slots[slot_loc].append(Insertion(match.pred))
    
    return slots


@dataclass(init=False)
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
    
    true: TranscriptionPath
    """The parsed ground truth with selected path (i. e. a selected
    option index in each multivariant block, if such blocks exist).
    """
    
    pred: SingleVariantTranscription
    """The parsed prediction, as passed into the constructor."""
    
    slots: dict[OuterLoc | InnerLoc, list[WORD_ERROR_TYPE]]
    """The predicted words which are packed into slots. Each slot
    represents a specific position in the ground truth: before, at or
    after a word in the ground truth. See :doc:`/guide_alignment_wer`
    for details.
    """

    _override_true_len_for_metrics: int | None = None
    # used for calculating WER only for highlighed words, see
    # `keep_errors_only_in_flagged_tokens_inplace` in evaluator.py
    
    def __init__(
        self,
        true: Transcription,
        pred: SingleVariantTranscription,
        absorb_insertions: bool = True,
    ):
        matches_list, multivariant_choices = (
            solve_optimal_alignment(true, pred)
        )
        true = true.select_single_path(multivariant_choices)
        slots = _matches_to_slots(true, matches_list.matches)
        if absorb_insertions:
            _absorb_insertions_into_replacements_inplace(true, slots)
        
        self.true = true
        self.pred = pred
        self.slots = dict(slots) # convert to dict to be serializable
    
    def error_listing(
        self,
        count_absorbed_insertions: bool = True,
        max_consecutive_insertions: int | None = None,
        skip_slots_with_zero_errors: bool = True,
    ) -> tuple[Metrics, dict[OuterLoc, ErrorListingElement]]:
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
        
        err_positions_dict: dict[OuterLoc, ErrorListingElement] = {}
        for loc, slot_values in self.slots.items():
            assert len(slot_values)
            loc_outer = OuterLoc(loc.mod, loc.pos)
            if not loc_outer in err_positions_dict:
                true = (
                    self.true.blocks[loc_outer.pos]
                    if loc_outer.mod == 'at'
                    else None
                )
                err_positions_dict[loc_outer] = ErrorListingElement(
                    outer_loc=loc_outer,
                    true=true,
                    true_text=true.to_text() if true is not None else None,
                    pred=[],
                    n_replacements=0,
                    n_insertions=0,
                    n_deletions=0,
                )
            
            err_pos = err_positions_dict[loc_outer]
            err_pos.pred += slot_values
            
            # count <Correct | Replacement | Insertion | Deletion> per cell
            slot_replacements = (
                sum(isinstance(x, Replacement) for x in slot_values)
            )
            slot_insertions = (
                sum(isinstance(x, Insertion) for x in slot_values)
            )
            slot_deletions = (
                sum(isinstance(x, Deletion) for x in slot_values)
            )
            slot_correct = (
                sum(isinstance(x, Correct) for x in slot_values)
            )
            
            # determine cell type
            has_replacements = slot_replacements > 0
            has_insertions = slot_insertions > 0
            has_deletions = slot_deletions > 0
            has_corrects = slot_correct > 0
            
            # validate cell type
            if has_replacements:
                # slot of "replacement" type, may only contain absorbed
                # insertions
                assert not has_deletions and not has_corrects
            else:
                # slot of "deletion", "insertion", or "correct" type
                assert sum((has_insertions, has_deletions, has_corrects)) == 1
        
            # count replacements, deletions
            err_pos.n_replacements += slot_replacements
            err_pos.n_deletions += slot_deletions
            
            # count insertions
            if has_replacements:
                # for "replacement" cells, we may count or not count absorbed
                # insertions
                if count_absorbed_insertions:
                    err_pos.n_insertions += slot_insertions
            elif max_consecutive_insertions is not None:
                # for "insertion" cell, set an upper bound for slot insertions
                # count
                err_pos.n_insertions += (
                    min(slot_insertions, max_consecutive_insertions)
                )
            else:
                err_pos.n_insertions += slot_insertions
        
        err_positions = {
            loc: pos
            for loc, pos in err_positions_dict.items()
            if pos.n_errors > 0 or not skip_slots_with_zero_errors
        }
        
        metrics = Metrics(
            true_len=(
                self.get_true_len()
                if self._override_true_len_for_metrics is None
                else self._override_true_len_for_metrics
            ),
            n_replacements=sum([
                pos.n_replacements for pos in err_positions.values()
            ]),
            n_insertions=sum([
                pos.n_insertions for pos in err_positions.values()
            ]),
            n_deletions=sum([
                pos.n_deletions for pos in err_positions.values()
            ]),
        )
        return metrics, err_positions

    def render_as_text(
        self,
        color_mode: Literal['ansi', 'html', None] = 'ansi',
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

        ma = MultipleAlignment(self.true, {name or 'pred': self})
        return ma.render_as_text(
            color_mode=color_mode,
            html_add_style=html_add_style,
            add_prediction_names=name is not None,
            max_cell_size=max_cell_size,
        )
          
    def get_true_len(self) -> int:
        """Get the words count in the ground truth.
        
        If there are multivariant blocks in the ground truth, selects
        the shortest (possibly empty) option in each block.
        
        Also, :class:`~asr_eval.align.transcription.Wildcard` tokens do
        not increment the ground truth length.
        """

        single_variant = _select_shortest_multi_variants(self.true.blocks)
        return _get_len(single_variant)


def _absorb_insertions_into_replacements_inplace(
    true: TranscriptionPath,
    slots: dict[OuterLoc | InnerLoc, list[WORD_ERROR_TYPE]]
):
    """Absorbs insertions into the neighbour blocks if this reduces CER.
    
    See the docstring for `Alignment(absorb_insertions=True)`.
    """
    
    slots_becoming_empty: set[OuterLoc | InnerLoc] = set()
    for slot_loc, slot_values in slots.items():
        if len(slot_values) == 1 and isinstance(slot_values[0], Replacement):
            true_text = cast(str, true.slot_to_token(slot_loc).value)
            text = cast(str, slot_values[0].token.value)
            
            # try to absorb insertions from the left side
            prev_slot = true.get_prev_slot(slot_loc)
            if prev_slot is not None and prev_slot in slots:
                prev_values = slots[prev_slot]
                for i in np.arange(len(prev_values))[::-1]:
                    if not isinstance(prev_values[i], Insertion):
                        break
                    prev_text = cast(str, prev_values[i].token.value) # type: ignore
                    if (
                        char_edit_distance(true_text, prev_text + ' ' + text)
                        < char_edit_distance(true_text, text)
                    ):
                        slot_values.insert(0, prev_values.pop())
                        text = prev_text + ' ' + text
                if len(prev_values) == 0:
                    slots_becoming_empty.add(prev_slot)
            
            # try to absorb insertions from the right side
            next_slot = true.get_next_slot(slot_loc)
            if next_slot is not None and next_slot in slots:
                next_values = slots[next_slot]
                for i in np.arange(len(next_values)):
                    if not isinstance(next_values[0], Insertion):
                        break
                    next_text = cast(str, next_values[0].token.value) # type: ignore
                    if (
                        char_edit_distance(true_text,  text + ' ' + next_text)
                        < char_edit_distance(true_text, text)
                    ):
                        slot_values.append(next_values.pop(0))
                        text = text + ' ' + next_text
                if len(next_values) == 0:
                    slots_becoming_empty.add(next_slot)
    
    for slot_loc in slots_becoming_empty:
        del slots[slot_loc]


_CHARWISE_RENDER = bool(int(os.environ.get('ASR_EVAL_CHARWISE_RENDER', '0')))


@dataclass
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
    
    baseline: Transcription
    """The baseline against which all the predictions are aligned.
    Usually a ground truth.
    """
    
    alignments: dict[str, Alignment]
    """The alignments against the
    :attr:`~asr_eval.align.alignment.MultipleAlignment.baseline`.
    """
    
    elapsed_times: dict[str, float] = field(default_factory=dict[str, float])
    """The inference times for each prediction (may be NaN or absent).
    """
    
    baseline_name: str | Literal['true'] = 'true'
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
        table = self.to_table()
        df = pd.DataFrame(
            columns=[  # type:ignore
                x.to_text() if x is not None else '<gap>'
                for x in get_outer_slots_values(self.baseline.blocks)
            ],
            data=table.to_numpy(),
        )
        df['name'] = list(self.alignments)
        df = df.set_index('name') # type: ignore
        return df
    
    def to_table(self) -> Table2D[list[WORD_ERROR_TYPE]]:
        """Convert into a table view, but using a better typed
        :class:`~asr_eval.utils.table.Table2D` instead of
        :code:`pd.DataFrame`, compared to
        :meth:`~asr_eval.align.alignment.MultipleAlignment.to_dataframe`.
        """
        outer_slots = list(get_outer_slots(self.baseline.blocks))
        outer_slot_to_index: dict[OuterLoc, int] = {
            loc: i for i, loc in enumerate(outer_slots)
        }
        
        outer_slot_values = Table2D[list[WORD_ERROR_TYPE]].construct(
            rows=len(self.alignments),
            cols=len(outer_slots),
            default=list,
        )

        for row_idx, alignment in enumerate(self.alignments.values()):
            for loc, values in alignment.slots.items():
                outer_loc = OuterLoc(loc.mod, loc.pos)
                outer_slot_idx = outer_slot_to_index[outer_loc]
                outer_slot_values[row_idx, outer_slot_idx] += values
        
        return outer_slot_values
        
    def render_as_text(
        self,
        color_mode: Literal['ansi', 'html', None] = 'ansi',
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
        unlabeled = self.baseline_name != 'true'
        
        table = self.to_table()
        
        # length of colored spans for Deletion()
        baseline_words = [
            self.baseline.text[block.start_pos:block.end_pos]
            if block is not None
            else ''
            for block in list(get_outer_slots_values(self.baseline.blocks))
        ]
        baseline_word_lengths = [len(w) for w in baseline_words]
        
        def _colorize_err(
            text: str,
            mode: Literal['ansi', 'html', None],
            type: Literal[
                'unlabeled', 'err', 'err_first', 'err_second', 'err_both'
            ] = 'err',
        ) -> tuple[str, int]:
            match mode:
                case None:
                    return text, len(text)
                case 'html':
                    match type:
                        case 'unlabeled':
                            color = '#FFDB85'
                        case 'err':
                            color = '#FF9C9C'
                        case 'err_first':
                            # second model is better
                            color = '#BAFFEF'
                        case 'err_second':
                            # baseline is better
                            color = '#FFE8B0'
                        case 'err_both':
                            # both baseline and second model made an error
                            color = '#C9C9C9'
                    return (
                        f'<span style="background-color: {color};">'
                        + text
                        + '</span>'
                    ), len(text)
                case 'ansi':
                    ansi_err_on_color = 'on_yellow'  # on_red is too dark
                    return colored(text, on_color=ansi_err_on_color), len(text)

        texts_list = list([al.pred for al in self.alignments.values()])
        
        def render_cell(
            row: int,
            col: int,
            cell: list[WORD_ERROR_TYPE],
        ) -> tuple[str, int]:
            # returns (text, text_len) tuple for a cell
            # text_len is a count of printable characters in the text,
            # excluding ANSI color codes and HTML tags
            nonlocal baseline_word_lengths, self
            words: list[str] = []
            lengths: list[int] = []
            for x in cell:
                p = '×' if charwise_mode else ' '
                text = (
                    p * max(1, baseline_word_lengths[col] // len(cell))
                    if isinstance(x, Deletion)
                    else texts_list[row] \
                        .text[x.token.start_pos:x.token.end_pos]
                )
                if max_cell_size is not None and len(text) > max_cell_size:
                    text = text[:max_cell_size] + '…'
                text_len = len(text)
                if not isinstance(x, Correct):
                    if unlabeled:
                        type = 'unlabeled'
                    elif table.shape[0] != 2:
                        type = 'err'
                    else:
                        # model1 vs model2 vs ground truth
                        cell_first = table[0, col]
                        cell_second = table[1, col]
                        first_has_errors = any(
                            not isinstance(x, Correct) for x in cell_first
                        )
                        second_has_errors = any(
                            not isinstance(x, Correct) for x in cell_second
                        )
                        both_have_errors = first_has_errors and second_has_errors
                        is_second = row == 1
                        if both_have_errors:
                            type = 'err_both'
                        elif is_second:
                            type = 'err_second'
                        else:
                            type = 'err_first'
                    text, text_len = _colorize_err(text, color_mode, type=type)
                words.append(text)
                lengths.append(text_len)
            return (
                ('' if charwise_mode else ' ').join(words),
                sum(lengths) + max(0, len(lengths) - 1)
            )
        
        # print(self.names)
        # print(self.table.to_pandas())

        # table_str keeps (text, text_len) tuple in each cell
        table_str: Table2D[tuple[str, int]] = (
            table.map_with_indices(render_cell)
        )
        
        first_row = list(zip(baseline_words, baseline_word_lengths))
        table_str.prepend_row(first_row)
        
        if add_prediction_names:
            table_str.prepend_col([
                ('|', 1)
                for _ in range(table_str.shape[0])
            ])
            table_str.prepend_col([
                (x, len(x))
                for x in [self.baseline_name] + list(self.alignments)
            ])

        col_lengths = [
            max([l for _, l in table_str[:, col_idx]])
            for col_idx in range(table_str.shape[1])
        ]

        lines: list[str] = []
        for row_idx in range(table_str.shape[0]):
            line = ('' if charwise_mode else ' ').join([
                text + ' ' * (length - text_len)
                for (text, text_len), length
                in zip(table_str[row_idx, :], col_lengths)
            ])
            lines.append(line)
            
        # use single quotes inside the string
        html_style = """white-space: pre; font-family: 'Consolas', \
            'Ubuntu Mono', 'Monaco', monospace"""

        match color_mode:
            case 'html':
                text_block = '<br/>'.join(lines)
                if html_add_style:
                    text_block = (
                        f'<span style="{html_style}">' + text_block + '</span>'
                    )
            case _:
                text_block = '\n'.join(lines)
        
        return text_block