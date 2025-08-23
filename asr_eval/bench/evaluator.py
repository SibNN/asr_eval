from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import chain
from typing import Literal

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
    error_listing_1: DataclassDataFrame[ErrorListingElement]
    error_listing_2: DataclassDataFrame[ErrorListingElement]
    # insertions_1: list[ErrorListingElement]
    # insertions_2: list[ErrorListingElement]
    # errors_in_1_both: list[ErrorListingElement]
    # errors_in_2_both: list[ErrorListingElement]
    # errors_in_1_but_not_2: list[ErrorListingElement]
    # errors_in_2_but_not_1: list[ErrorListingElement]


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
    ) -> DatasetPipelinePairComparison:
        err_positions_1: list[ErrorListingElement] = []
        err_positions_2: list[ErrorListingElement] = []
        
        for sample_idx, sample in enumerate(dataset_data.samples):
            if pipeline_name_1 in sample.pipelines and pipeline_name_2 in sample.pipelines:
                for pos in sample.pipelines[pipeline_name_1].err_positions:
                    err_positions_1.append(replace(pos, sample_idx=sample_idx))
                for pos in sample.pipelines[pipeline_name_2].err_positions:
                    err_positions_2.append(replace(pos, sample_idx=sample_idx))
        
        return DatasetPipelinePairComparison(
            pipeline_name_1=pipeline_name_1,
            pipeline_name_2=pipeline_name_2,
            error_listing_1=DataclassDataFrame[ErrorListingElement](err_positions_1),
            error_listing_2=DataclassDataFrame[ErrorListingElement](err_positions_2),
        )
        
        # result = DatasetPipelinePairComparison(
        #     pipeline_name_1=pipeline_name_1,
        #     pipeline_name_2=pipeline_name_2,
        #     insertions_1=[],
        #     insertions_2=[],
        #     errors_in_1_both=[],
        #     errors_in_2_both=[],
        #     errors_in_1_but_not_2=[],
        #     errors_in_2_but_not_1=[],
        # )
        
        # for sample_idx, sample in enumerate(dataset_data.samples):
        #     if pipeline_name_1 in sample.pipelines and pipeline_name_2 in sample.pipelines:
                
        #         err_positions_1 = sample.pipelines[pipeline_name_1].err_positions
        #         err_positions_2 = sample.pipelines[pipeline_name_2].err_positions
                
        #         # 1. for outer "pre" positions (insertions)
                
        #         pre_1 = [pos for pos in err_positions_1 if pos.outer_loc[0] == 'pre']
        #         pre_2 = [pos for pos in err_positions_2 if pos.outer_loc[0] == 'pre']
                
        #         for pos in pre_1:
        #             result.insertions_1.append(replace(pos, sample_idx=sample_idx))
        #         for pos in pre_2:
        #             result.insertions_2.append(replace(pos, sample_idx=sample_idx))
                
        #         # 2. for outer "at" positions (insertions)
                
        #         at_1 = [pos for pos in err_positions_1 if pos.outer_loc[0] == 'at']
        #         at_2 = [pos for pos in err_positions_2 if pos.outer_loc[0] == 'at']
                
        #         locs_1 = set([pos.outer_loc for pos in at_1])
        #         locs_2 = set([pos.outer_loc for pos in at_2])
                
        #         for pos in at_1:
        #             if pos.outer_loc in locs_2:
        #                 result.errors_in_1_both.append(replace(pos, sample_idx=sample_idx))
        #             else:
        #                 result.errors_in_1_but_not_2.append(replace(pos, sample_idx=sample_idx))
        #         for pos in at_2:
        #             if pos.outer_loc in locs_1:
        #                 result.errors_in_2_both.append(replace(pos, sample_idx=sample_idx))
        #             else:
        #                 result.errors_in_2_but_not_1.append(replace(pos, sample_idx=sample_idx))
                
        # return result