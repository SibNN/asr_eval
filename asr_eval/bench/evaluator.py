from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class SampleData:
    sample_idx: int
    baseline_transcription_html: str
    baseline_is_ground_truth: bool
    pipelines: dict[str, SamplePipelineData]
    baseline_name: str = ''


@dataclass
class SamplePipelineData:
    n_errors: int
    n_replacements: int
    n_insertions: int
    n_deletions: int
    elapsed_time: float
    transcription_html: str


class Evaluator(PredictionLoader):
    def get_dataset_data(
        self,
        dataset_name: str,
        pipeline_names: list[str],
        count_absorbed_insertions: bool = True,
        max_consecutive_insertions: int | None = None,
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
                    transcription_html=aligned_transcription,
                    elapsed_time=pred.elapsed_time,
                    n_errors=sample_metrics.n_errors,
                    n_replacements=sample_metrics.n_replacements,
                    n_insertions=sample_metrics.n_insertions,
                    n_deletions=sample_metrics.n_deletions,
                )
            
            samples.append(SampleData(
                sample_idx=sample_idx,
                baseline_transcription_html=aligned_html[0],
                baseline_is_ground_truth=baseline_is_ground_truth,
                pipelines=pipelines,
                baseline_name=str(multiple_alignment.baseline_name),
            ))
        
        return DatasetData(samples=samples)