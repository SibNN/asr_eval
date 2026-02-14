from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, replace
from itertools import chain
import textwrap
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from plotly.graph_objs._figure import Figure

from asr_eval.align.transcription import OuterLoc, Wildcard
from asr_eval.align.alignment import (
    Alignment, ErrorListingElement, MultipleAlignment
)
from asr_eval.align.metrics import DatasetMetric, Metrics


__all__ = [
    'get_dataset_data',
    'DatasetData',
    'SampleData',
    'SamplePipelineData',
    'compare_pipelines',
]


@dataclass
class DatasetData:
    """Output format for the
    :class:`~asr_eval.bench.evaluator.get_dataset_data` function.
    """
    
    samples: list[SampleData]
    """A list of :class:`~asr_eval.bench.evaluator.SampleData` for
    all the sample ids for which we have at least one prediction.
    """
    
    full_samples: list[int]
    """A list of sample ids for which all the pipelines have a
    prediction. These sample ids are used for averaging metrics, to
    avoid a problem where different pipeline predicions are averaged
    across differen samples set, and hence are not directly comparable.
    """
    
    dataset_metric: dict[str, DatasetMetric]
    """Metrics for each pipeline, averaged across
    :attr:`~asr_eval.bench.evaluator.DatasetData.full_samples`,
    if the :code:`full_samples` list is not empty.
    """
    
    def get_all_pipelines(self) -> list[str]:
        """Get all the pipelines for which we have at least one
        prediction.
        """
        return list(set(chain(*[list(s.pipelines) for s in self.samples])))
    

@dataclass
class DatasetPipelinePairComparison:
    """The result of the
    :func:`~asr_eval.bench.evaluator.compare_pipelines`.
    
    To be documented.
    """
    pipeline_name_1: str
    pipeline_name_2: str
    errs_1_shared: list[ErrorListingElement]
    errs_2_shared: list[ErrorListingElement]
    errs_1_unique_insertions: list[ErrorListingElement]
    errs_2_unique_insertions: list[ErrorListingElement]
    unique_replacements_top: list[str]
    errs_1_unique_replacements_top: list[list[ErrorListingElement]]
    errs_2_unique_replacements_top: list[list[ErrorListingElement]]
    errs_1_unique_replacements_other: list[ErrorListingElement]
    errs_2_unique_replacements_other: list[ErrorListingElement]
    
    def counts(self) -> list[tuple[str, int, int, str, str]]:
        result: list[tuple[str, int, int, str, str]] = [
            (
                'shared_errors',
                sum([pos.n_errors for pos in self.errs_1_shared]),
                sum([pos.n_errors for pos in self.errs_2_shared]),
                '#IDX TRUE PRED: ' + ', '.join([
                    f'#{pos.sample_id} T={str(pos.true_text)}'
                    f' P={pos.pred_text}'
                    for pos in self.errs_1_shared
                ]),
                '#IDX TRUE PRED: ' + ', '.join([
                    f'#{pos.sample_id} T={str(pos.true_text)}'
                    f' P={pos.pred_text}'
                    for pos in self.errs_2_shared
                ]),
            ),
            (
                'insertions',
                sum([pos.n_errors for pos in self.errs_1_unique_insertions]),
                sum([pos.n_errors for pos in self.errs_2_unique_insertions]),
                '#IDX PRED: ' + ', '.join([
                    f'#{pos.sample_id} {pos.pred_text}'
                    for pos in self.errs_1_unique_insertions
                ]),
                '#IDX PRED: ' + ', '.join([
                    f'#{pos.sample_id} {pos.pred_text}'
                    for pos in self.errs_2_unique_insertions
                ]),
            ),
            (
                'other_unique',
                sum([
                    pos.n_errors
                    for pos in self.errs_1_unique_replacements_other
                ]),
                sum([
                    pos.n_errors
                    for pos in self.errs_2_unique_replacements_other
                ]),
                '#IDX TRUE PRED: ' + ', '.join([
                    f'#{pos.sample_id} T={str(pos.true_text)}'
                    f' P={pos.pred_text}'
                    for pos in self.errs_1_unique_replacements_other
                ]),
                '#IDX TRUE PRED: ' + ', '.join([
                    f'#{pos.sample_id} T={str(pos.true_text)}'
                    f' P={pos.pred_text}'
                    for pos in self.errs_2_unique_replacements_other
                ]),
            ),
        ]
        
        for word, listing_1, listing_2 in zip(
            self.unique_replacements_top,
            self.errs_1_unique_replacements_top,
            self.errs_2_unique_replacements_top,
        ):
            result.append((
                '"' + word + '"',
                sum([pos.n_errors for pos in listing_1]),
                sum([pos.n_errors for pos in listing_2]),
                '#IDX TRUE PRED: ' + ', '.join([
                    f'#{pos.sample_id} T={str(pos.true_text)}'
                    f' P={pos.pred_text}'
                    for pos in listing_1
                ]),
                '#IDX TRUE PRED: ' + ', '.join([
                    f'#{pos.sample_id} T={str(pos.true_text)}'
                    f' P={pos.pred_text}'
                    for pos in listing_2
                ]),
            ))
        
        return result
    
    def plot(self) -> Figure:
        import plotly.express as px

        counts = self.counts()
        labels = [
            f'{label} ({count1} vs {count2})'
            for label, count1, count2, _desc1, _desc2 in counts
        ]
        counts1 = [
            count1
            for _label, count1, _count2, _desc1, _desc2 in counts
        ]
        counts2 = [
            count2
            for _label, _count1, count2, _desc1, _desc2 in counts
        ]
        
        def wrap_text(text: str) -> str:
            lines = textwrap.wrap(text, width=150)
            if len(lines) > 10:
                # otherwise plotly does not show any text at all
                lines = lines[:10] + [f'... + {len(lines) - 10} rows hidden']
            return '<br>'.join(lines)
        
        desc1 = [
            wrap_text(desc1)
            for _label, _count1, _count2, desc1, _desc2 in counts
        ]
        desc2 = [
            wrap_text(desc2)
            for _label, _count1, _count2, _desc1, desc2 in counts
        ]
        
        df = pd.concat([
            pd.DataFrame({
                'pipeline': self.pipeline_name_2,
                'type': labels,
                'n_errs': counts2,
                'details': desc2,
            }),
            pd.DataFrame({
                'pipeline': self.pipeline_name_1,
                'type': labels,
                'n_errs': counts1,
                'details': desc1,
            }),
        ])

        fig = px.bar( # type: ignore
            df,
            y="pipeline",
            x="n_errs",
            color="type",
            hover_data='details',
            # width=100,
            # height=250,
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        fig.update_traces(width=0.3) # type: ignore
        fig.update_layout( # type: ignore
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            yaxis_title=None,
            xaxis_title=None,
        )
        return fig
    


@dataclass
class SampleData:
    sample_id: int
    baseline_transcription_html: str | None
    baseline_is_ground_truth: bool
    pipelines: dict[str, SamplePipelineData]
    baseline_name: str = ''


@dataclass
class SamplePipelineData:
    """A field of the
    :class:`~asr_eval.bench.evaluator.DatasetData` dataclass, represents
    the :class:`~asr_eval.align.alignment.Alignment` between ground
    truth and prediction, as well as other useful information.
    """
    
    err_positions: dict[OuterLoc, ErrorListingElement]
    """The output of
    :meth:`~asr_eval.align.alignment.Alignment.error_listing`
    """
    
    metrics: Metrics
    """The output of
    :meth:`~asr_eval.align.alignment.Alignment.error_listing`
    """
    
    elapsed_time: float
    """Inference time, may be NaN if not known."""
    
    transcription_html: str | None
    """The aligned transcription in HTML to display."""
    
    alignment: Alignment
    """The alignment between ground truth and prediction"""


def _has_digits(multiple_alignment: MultipleAlignment)-> bool:
    for token in multiple_alignment.baseline.list_all_tokens():
        if (
            not isinstance(token.value, Wildcard)
            and any(char.isdigit() for char in token.value)
        ):
            return True
    for alignment in multiple_alignment.alignments.values():
        for token in alignment.pred.list_all_tokens():
            if (
                not isinstance(token.value, Wildcard)
                and any(char.isdigit() for char in token.value)
            ):
                return True
    return False


def get_dataset_data(
    multiple_alignments: dict[int, MultipleAlignment],
    count_absorbed_insertions: bool = True,
    max_consecutive_insertions: int | None = None,
    wer_averaging_mode: Literal['plain', 'concat'] = 'concat',
    exclude_samples_with_digits: bool = False,
    max_samples_to_render: int | None = None,
) -> DatasetData:
    """Takes raw multiple alignments (usually from
    :meth`~asr_eval.bench.loader.PredictionLoader.get_multiple_alignments`)
    and 1) renders multiple alignments in a displayable form, 2) averages
    metrics across all samples.
    
    Acts as a main utility for the ASR dashboard data model.
    
    See Also:
        More details and examples in the user guide
        :doc:`/guide_alignment_wer`.
    
    Args:
        multiple_alignments: multiple alignments for several sample ids
            in some dataset. All the multiple alignments should NOT
            necessary contain the same set of pipelines.
        count_absorbed_insertions: a parameter for
            :meth:`~asr_eval.align.alignment.Alignment.error_listing`
            when calculating metrics.
        max_consecutive_insertions: a parameter for
            :meth:`~asr_eval.align.alignment.Alignment.error_listing`
            when calculating metrics.
        wer_averaging_mode: a parameter for
            :meth:`~asr_eval.align.metrics.DatasetMetric.from_samples`
            when averagint metrics.
        exclude_samples_with_digits: if True, when averagint metrics,
            excludes all samples where a digit is found either in the
            ground truth transcription, or in some of the pipeline
            predictions. This acts as a "poor man's solution" to avoid
            issues with normalization of numericals.
        max_samples_to_render: if not None, don't render multiple
            alignments for all samples except the specified number of
            samples.
    
    Returns:
        See the :class:`~asr_eval.bench.evaluator.DatasetData` docs.
    """
    
    from joblib import Parallel, delayed # pyright: ignore[reportUnknownVariableType]

    samples: list[SampleData] = []
    for sample_id, multiple_alignment in multiple_alignments.items():
        if exclude_samples_with_digits and _has_digits(multiple_alignment):
            continue
        
        should_render = (
            max_samples_to_render is None
            or len(samples) < max_samples_to_render
        )
        # sort_pipelines=True is important for correct coloring when 2
        # pipelines
        baseline_is_ground_truth = multiple_alignment.baseline_name == 'true'
        
        if should_render:
            aligned_html = (  # header (ground truth) + pipelines
                multiple_alignment
                .render_as_text(
                    color_mode='html',
                    html_add_style=False,
                    add_prediction_names=False,
                )
                .split('<br/>')
            )
            header_html, pipelines_html = aligned_html[0], aligned_html[1:]
        else:
            header_html, pipelines_html = None, None
        
        pipelines: dict[str, SamplePipelineData] = {}
        for pipeline_idx, (pipeline_name, alignment) in (
            enumerate(multiple_alignment.alignments.items())
        ):
            sample_metrics, err_positions = alignment.error_listing(
                max_consecutive_insertions=max_consecutive_insertions,
                count_absorbed_insertions=count_absorbed_insertions,
            )
            pipelines[pipeline_name] = SamplePipelineData(
                err_positions=err_positions,
                metrics=sample_metrics,
                transcription_html=(
                    pipelines_html[pipeline_idx]
                    if pipelines_html is not None
                    else None
                ),
                elapsed_time=(
                    multiple_alignment.elapsed_times
                    .get(pipeline_name, np.nan)
                ),
                alignment=alignment,
            )
        
        samples.append(SampleData(
            sample_id=sample_id,
            baseline_transcription_html=header_html,
            baseline_is_ground_truth=baseline_is_ground_truth,
            pipelines=pipelines,
            baseline_name=multiple_alignment.baseline_name,
        ))
    
    all_pipelines = set(chain(*[sample.pipelines for sample in samples]))
    full_sample_indices = [
        i for i, sample in enumerate(samples)
        if set(sample.pipelines) == all_pipelines
    ]
    full_samples = [
        sample for i, sample in enumerate(samples)
        if i in full_sample_indices
    ]
    
    pipeline_metrics = cast(list[DatasetMetric], Parallel(n_jobs=-1)(
        delayed(DatasetMetric.from_samples)(
            samples=[
                sample.pipelines[pipeline_name].metrics
                for sample in full_samples
            ],
            wer_averaging_mode=wer_averaging_mode,
        )
        for pipeline_name in all_pipelines
    ))
    
    dataset_metric = dict(sorted(
        zip(all_pipelines, pipeline_metrics),
        key=lambda item: item[1].wer.main_value
    ))
    
    return DatasetData(
        samples=samples,
        full_samples=full_sample_indices,
        dataset_metric=dataset_metric,
    )


def compare_pipelines(
    dataset_data: DatasetData,
    pipeline_name_1: str,
    pipeline_name_2: str,
) -> DatasetPipelinePairComparison:
    """An utility for fine-grained comparison of two pipelines on the
    same dataset.
    
    To be documented.
    """
    err_positions_1: list[ErrorListingElement] = []
    err_positions_2: list[ErrorListingElement] = []
    
    for sample in dataset_data.samples:
        if (
            pipeline_name_1 in sample.pipelines
            and pipeline_name_2 in sample.pipelines
        ):
            for pos in (
                sample.pipelines[pipeline_name_1].err_positions.values()
            ):
                err_positions_1.append(
                    replace(pos, sample_id=sample.sample_id)
                )
            for pos in (
                sample.pipelines[pipeline_name_2].err_positions.values()
            ):
                err_positions_2.append(
                    replace(pos, sample_id=sample.sample_id)
                )
    
    at_locs_1 = set([
        (cast(int, pos.sample_id), pos.outer_loc)
        for pos in err_positions_1
        if pos.outer_loc.mod == 'at'
    ])
    at_locs_2 = set([
        (cast(int, pos.sample_id), pos.outer_loc)
        for pos in err_positions_2
        if pos.outer_loc.mod == 'at'
    ])
    both_errors = at_locs_1.intersection(at_locs_2)
    
    errs_1_shared = [
        pos for pos in err_positions_1
        if (pos.sample_id, pos.outer_loc) in both_errors
    ]
    errs_2_shared = [
        pos for pos in err_positions_2
        if (pos.sample_id, pos.outer_loc) in both_errors
    ]
    errs_1_unique = [
        pos for pos in err_positions_1
        if (pos.sample_id, pos.outer_loc) not in both_errors
    ]
    errs_2_unique = [
        pos for pos in err_positions_2
        if (pos.sample_id, pos.outer_loc) not in both_errors
    ]
    errs_1_unique_insertions = [
        pos for pos in errs_1_unique if pos.outer_loc.mod == 'pre'
    ]
    errs_2_unique_insertions = [
        pos for pos in errs_2_unique if pos.outer_loc.mod == 'pre'
    ]
    errs_1_unique_replacements = [
        pos for pos in errs_1_unique if pos.outer_loc.mod == 'at'
    ]
    errs_2_unique_replacements = [
        pos for pos in errs_2_unique if pos.outer_loc.mod == 'at'
    ]
    
    texts_and_counts_1 = Counter(sorted([
        cast(str, pos.true_text) for pos in errs_1_unique_replacements
    ]))
    texts_and_counts_2 = Counter(sorted([
        cast(str, pos.true_text) for pos in errs_1_unique_replacements
    ]))
    texts_and_counts = texts_and_counts_1 | texts_and_counts_2
    
    max_top_words = 100
    min_count = 2
    
    top_texts: list[str] = []
    for text, count in texts_and_counts.most_common(max_top_words):
        if count < min_count:
            break
        top_texts.append(text)
    
    errs_1_unique_replacements_top = [
        [pos for pos in errs_1_unique_replacements if pos.true_text == text]
        for text in top_texts
    ]
    errs_2_unique_replacements_top = [
        [pos for pos in errs_2_unique_replacements if pos.true_text == text]
        for text in top_texts
    ]
    errs_1_unique_replacements_other = [
        pos for pos in errs_1_unique_replacements
        if pos.true_text not in top_texts
    ]
    errs_2_unique_replacements_other = [
        pos for pos in errs_2_unique_replacements
        if pos.true_text not in top_texts
    ]
    
    return DatasetPipelinePairComparison(
        pipeline_name_1=pipeline_name_1,
        pipeline_name_2=pipeline_name_2,
        errs_1_shared=errs_1_shared,
        errs_2_shared=errs_2_shared,
        errs_1_unique_insertions=errs_1_unique_insertions,
        errs_2_unique_insertions=errs_2_unique_insertions,
        unique_replacements_top=top_texts,
        errs_1_unique_replacements_top=errs_1_unique_replacements_top,
        errs_2_unique_replacements_top=errs_2_unique_replacements_top,
        errs_1_unique_replacements_other=errs_1_unique_replacements_other,
        errs_2_unique_replacements_other=errs_2_unique_replacements_other,
    )