from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Literal

from ..utils.dataframe import DataclassDataFrame
from ..align.transcription import OUTER_LOC, MultiVariantBlock, Token
from ..align.alignment import SLOT_VALUES, Alignment, Correct
from ..align.metrics import DatasetMetric, Metrics
from .loader import PredictionLoader


__all__ = [
    'DatasetData',
    'SampleData',
    'SamplePipelineData',
    'Evaluator',
]


@dataclass
class DatasetData:
    samples: list[SampleData]
    full_samples: list[int]
    dataset_metric: dict[str, DatasetMetric]
    
    def get_all_pipelines(self) -> list[str]:
        return list(set(chain(*[list(s.pipelines) for s in self.samples])))
    

@dataclass
class DatasetPipelinePairComparison:
    pipeline_name_1: str
    pipeline_name_2: str
    errors_1_but_not_2: list[tuple[str, list[UnevenError]]]
    errors_2_but_not_1: list[tuple[str, list[UnevenError]]]


@dataclass
class SampleData:
    sample_idx: int
    baseline_transcription_html: str
    baseline_is_ground_truth: bool
    pipelines: dict[str, SamplePipelineData]
    baseline_name: str = ''


@dataclass
class SamplePipelineData:
    metrics: Metrics
    elapsed_time: float
    transcription_html: str
    alignment: Alignment


@dataclass
class UnevenError:
    '''
    An outer slot when only one of two models made a mistake. Typically deletions
    or replacements, insertions may be included only if part of a multivariant block.
    
    TODO better docstring
    '''
    sample_idx: int
    outer_loc: OUTER_LOC
    true: Token | MultiVariantBlock
    true_text: str
    pred: SLOT_VALUES


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
                sample_metrics = alignment.metric_summary(
                    count_absorbed_insertions=count_absorbed_insertions,
                    max_consecutive_insertions=max_consecutive_insertions
                )
                pipelines[pipeline_name] = SamplePipelineData(
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
        
        # for pipeline_name, pipeline_dataset_metric in dataset_metric.items():
        #     print(pipeline_name, pipeline_dataset_metric.wer.main_value)
        
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
    ) -> DatasetPipelinePairComparison:
        errors_1_but_not_2: list[UnevenError] = []
        errors_2_but_not_1: list[UnevenError] = []
        for sample_idx, sample in enumerate(dataset_data.samples):
            if pipeline_name_1 in sample.pipelines and pipeline_name_2 in sample.pipelines:
                alignment_1 = sample.pipelines[pipeline_name_1].alignment
                alignment_2 = sample.pipelines[pipeline_name_2].alignment
                true = alignment_1.true
                
                outer_values_1 = alignment_1.to_outer_slots()
                outer_values_2 = alignment_2.to_outer_slots()
                outer_locs = set(outer_values_1).intersection(set(outer_values_2))
                
                for outer_loc in outer_locs:
                    outer_mod, outer_idx = outer_loc
                    if outer_mod == 'at':
                        true_block = true.tokens[outer_idx]
                        values_1 = outer_values_1[outer_loc]
                        values_2 = outer_values_2[outer_loc]
                        has_errors_1 = any(not isinstance(x, Correct) for x in values_1)
                        has_errors_2 = any(not isinstance(x, Correct) for x in values_2)
                        if has_errors_1 and not has_errors_2:
                            errors_1_but_not_2.append(UnevenError(
                                sample_idx=sample_idx,
                                outer_loc=outer_loc,
                                true=true_block,
                                true_text=true_block.to_text(),
                                pred=values_1,
                            ))
                        elif has_errors_2 and not has_errors_1:
                            errors_2_but_not_1.append(UnevenError(
                                sample_idx=sample_idx,
                                outer_loc=outer_loc,
                                true=true_block,
                                true_text=true_block.to_text(),
                                pred=values_2,
                            ))
        
        return DatasetPipelinePairComparison(
            pipeline_name_1=pipeline_name_1,
            pipeline_name_2=pipeline_name_2,
            errors_1_but_not_2=_uneven_errors_to_sorted_groups(errors_1_but_not_2),
            errors_2_but_not_1=_uneven_errors_to_sorted_groups(errors_2_but_not_1),
        )


def _uneven_errors_to_sorted_groups(
    errors: list[UnevenError]
) -> list[tuple[str, list[UnevenError]]]:
    groups = DataclassDataFrame[UnevenError](errors).groupby('true_text')
    top_groups = sorted(groups, key=lambda item: -len(item[1].data))
    return [(true_text, group.data) for true_text, group in top_groups]