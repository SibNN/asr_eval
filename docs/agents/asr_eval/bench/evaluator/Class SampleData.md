# Class SampleData (defined in asr_eval/bench/evaluator.py at lines 226-233)

@dataclasses.dataclass
class SampleData:
...

    sample_id: int

    baseline_transcription_html: str | None

    baseline_is_ground_truth: bool

    pipelines: dict[str, asr_eval.bench.evaluator.SamplePipelineData]

    baseline_name: str = ''