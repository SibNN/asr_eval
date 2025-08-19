from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Literal

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