from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from itertools import chain
from typing import Literal, cast

import pandas as pd
import plotly.express as px
from plotly.graph_objs._figure import Figure

from ..utils.dataframe import DataclassDataFrame
from ..align.alignment import Alignment, ErrorListingElement
from ..align.metrics import DatasetMetric, Metrics
from .loader import PredictionLoader


__all__ = [
    'DatasetData',
    'SampleData',
    'SamplePipelineData',
    'Evaluator',
]


SAMPLE_IDX = int


def group_by_true_text(
    errors: list[ErrorListingElement]
) -> list[tuple[str, list[ErrorListingElement]]]:
    groups = DataclassDataFrame[ErrorListingElement](errors).groupby('true_text')
    top_groups = sorted(groups, key=lambda item: -len(item[1].data))
    return [(true_text, group.data) for true_text, group in top_groups]


@dataclass
class DatasetData:
    samples: list[SampleData]
    full_samples: list[SAMPLE_IDX]
    dataset_metric: dict[str, DatasetMetric]
    
    def get_all_pipelines(self) -> list[str]:
        return list(set(chain(*[list(s.pipelines) for s in self.samples])))
    

@dataclass
class DatasetPipelinePairComparison:
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
    
    def counts(self) -> list[tuple[str, int, int]]:
        result: list[tuple[str, int, int]] = [
            (
                'shared_errors',
                sum([pos.n_errors for pos in self.errs_1_shared]),
                sum([pos.n_errors for pos in self.errs_2_shared]),
            ),
            (
                'insertions',
                sum([pos.n_errors for pos in self.errs_1_unique_insertions]),
                sum([pos.n_errors for pos in self.errs_2_unique_insertions]),
            ),
            (
                'other_unique',
                sum([pos.n_errors for pos in self.errs_1_unique_replacements_other]),
                sum([pos.n_errors for pos in self.errs_2_unique_replacements_other]),
            ),
        ]
        
        for word, listing_1, listing_2 in zip(
            self.unique_replacements_top,
            self.errs_1_unique_replacements_top,
            self.errs_2_unique_replacements_top,
        ):
            result.append((
                word,
                sum([pos.n_errors for pos in listing_1]),
                sum([pos.n_errors for pos in listing_2]),
            ))
        
        return result
    
    def plot(self) -> Figure:
        counts = self.counts()
        labels = [_label for _label, _count1, _count2 in counts]
        counts1 = [_count1 for _label, _count1, _count2 in counts]
        counts2 = [_count2 for _label, _count1, _count2 in counts]
        
        df = pd.concat([
            pd.DataFrame({'pipeline': 'pipeline 1', 'type': labels, 'n_errs': counts1}),
            pd.DataFrame({'pipeline': 'pipeline 2', 'type': labels, 'n_errs': counts2}),
        ])

        fig = px.bar(df, y="pipeline", x="n_errs", color="type", width=1000, height=250) # type: ignore
        return fig
    


@dataclass
class SampleData:
    sample_idx: SAMPLE_IDX
    baseline_transcription_html: str
    baseline_is_ground_truth: bool
    pipelines: dict[str, SamplePipelineData]
    baseline_name: str = ''


@dataclass
class SamplePipelineData:
    err_positions: list[ErrorListingElement]
    metrics: Metrics
    elapsed_time: float
    transcription_html: str
    alignment: Alignment


class Evaluator(PredictionLoader):
    def get_dataset_data(
        self,
        dataset_name: str,
        pipeline_names: list[str],
        count_absorbed_insertions: bool = True,
        max_consecutive_insertions: int | None = None,
        wer_averaging_mode: Literal['plain', 'concat'] = 'concat',
    ) -> DatasetData:
        multiple_alignments = self.get_multiple_alignments(dataset_name, pipeline_names)
        
        samples: list[SampleData] = []
        for sample_idx, multiple_alignment in multiple_alignments.items():
            baseline_is_ground_truth = multiple_alignment.baseline_name is True
            aligned_html = (
                multiple_alignment
                .view()
                .render_as_text(mode='html', html_add_style=False, add_pipeline_names=False)
                .split('<br/>')
            )
            
            pipelines: dict[str, SamplePipelineData] = {}
            for (pipeline_name, alignment), aligned_transcription in zip(
                multiple_alignment.alignments.items(), aligned_html[1:]
            ):
                pred = self.get_prediction(dataset_name, sample_idx, pipeline_name)
                err_positions, sample_metrics = alignment.error_listing(
                    count_absorbed_insertions=count_absorbed_insertions,
                    max_consecutive_insertions=max_consecutive_insertions
                )
                pipelines[pipeline_name] = SamplePipelineData(
                    err_positions=err_positions,
                    metrics=sample_metrics,
                    transcription_html=aligned_transcription,
                    elapsed_time=pred.elapsed_time,
                    alignment=alignment,
                )
            
            samples.append(SampleData(
                sample_idx=sample_idx,
                baseline_transcription_html=aligned_html[0],
                baseline_is_ground_truth=baseline_is_ground_truth,
                pipelines=pipelines,
                baseline_name=str(multiple_alignment.baseline_name),
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
        
        dataset_metric = {
            pipeline_name: DatasetMetric.from_samples(
                samples=[sample.pipelines[pipeline_name].metrics for sample in full_samples],
                wer_averaging_mode=wer_averaging_mode,
            )
            for pipeline_name in all_pipelines
        }
        
        dataset_metric = dict(sorted(
            dataset_metric.items(),
            key=lambda item: item[1].wer.main_value
        ))
        
        return DatasetData(
            samples=samples,
            full_samples=full_sample_indices,
            dataset_metric=dataset_metric,
        )
    
    @staticmethod
    def compare_pipelines(
        dataset_data: DatasetData,
        pipeline_name_1: str,
        pipeline_name_2: str,
        max_top_words: int = 100,
    ) -> DatasetPipelinePairComparison:
        err_positions_1: list[ErrorListingElement] = []
        err_positions_2: list[ErrorListingElement] = []
        
        for sample_idx, sample in enumerate(dataset_data.samples):
            if pipeline_name_1 in sample.pipelines and pipeline_name_2 in sample.pipelines:
                for pos in sample.pipelines[pipeline_name_1].err_positions:
                    err_positions_1.append(replace(pos, sample_idx=sample_idx))
                for pos in sample.pipelines[pipeline_name_2].err_positions:
                    err_positions_2.append(replace(pos, sample_idx=sample_idx))
        
        at_locs_1 = set([
            (cast(int, pos.sample_idx), pos.outer_loc)
            for pos in err_positions_1
            if pos.outer_loc[0] == 'at'
        ])
        at_locs_2 = set([
            (cast(int, pos.sample_idx), pos.outer_loc)
            for pos in err_positions_2
            if pos.outer_loc[0] == 'at'
        ])
        both_errors = at_locs_1.intersection(at_locs_2)
        
        errs_1_shared = [
            pos for pos in err_positions_1
            if (pos.sample_idx, pos.outer_loc) in both_errors
        ]
        errs_2_shared = [
            pos for pos in err_positions_2
            if (pos.sample_idx, pos.outer_loc) in both_errors
        ]
        errs_1_unique = [
            pos for pos in err_positions_1
            if (pos.sample_idx, pos.outer_loc) not in both_errors
        ]
        errs_2_unique = [
            pos for pos in err_positions_2
            if (pos.sample_idx, pos.outer_loc) not in both_errors
        ]
        errs_1_unique_insertions = [pos for pos in errs_1_unique if pos.outer_loc[0] == 'pre']
        errs_2_unique_insertions = [pos for pos in errs_2_unique if pos.outer_loc[0] == 'pre']
        errs_1_unique_replacements = [pos for pos in errs_1_unique if pos.outer_loc[0] == 'at']
        errs_2_unique_replacements = [pos for pos in errs_2_unique if pos.outer_loc[0] == 'at']
        
        texts_and_counts = Counter(sorted(
            [cast(str, pos.true_text) for pos in errs_1_unique_replacements]
            + [cast(str, pos.true_text) for pos in errs_2_unique_replacements]
        ))
        
        unique_replacements_top_texts: list[str] = []
        for text, count in texts_and_counts.most_common(max_top_words):
            if count < 2:
                break
            unique_replacements_top_texts.append(text)
        
        unique_replacements_top = [text for text in unique_replacements_top_texts]
        errs_1_unique_replacements_top = [
            [pos for pos in errs_1_unique_replacements if pos.true_text == text]
            for text in unique_replacements_top_texts
        ]
        errs_2_unique_replacements_top = [
            [pos for pos in errs_2_unique_replacements if pos.true_text == text]
            for text in unique_replacements_top_texts
        ]
        errs_1_unique_replacements_other = [
            pos for pos in errs_1_unique_replacements
            if pos.true_text not in unique_replacements_top
        ]
        errs_2_unique_replacements_other = [
            pos for pos in errs_2_unique_replacements
            if pos.true_text not in unique_replacements_top
        ]
        
        return DatasetPipelinePairComparison(
            pipeline_name_1=pipeline_name_1,
            pipeline_name_2=pipeline_name_2,
            errs_1_shared=errs_1_shared,
            errs_2_shared=errs_2_shared,
            errs_1_unique_insertions=errs_1_unique_insertions,
            errs_2_unique_insertions=errs_2_unique_insertions,
            unique_replacements_top=unique_replacements_top,
            errs_1_unique_replacements_top=errs_1_unique_replacements_top,
            errs_2_unique_replacements_top=errs_2_unique_replacements_top,
            errs_1_unique_replacements_other=errs_1_unique_replacements_other,
            errs_2_unique_replacements_other=errs_2_unique_replacements_other,
        )