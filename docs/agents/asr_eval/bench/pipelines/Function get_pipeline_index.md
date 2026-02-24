# Function get_pipeline_index (defined in asr_eval/bench/pipelines/_registry.py at lines 130-140)

@functools.cache
def get_pipeline_index(name: str) -> int:
    """Get an index (in registration order) for a registered pipeline,
    or -1 if not registered.
    """
    ...