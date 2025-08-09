import argparse

import librosa

from ..bench.pipelines import TimedTranscriberPipeline, TranscriberPipeline, get_pipeline
from ..utils.types import FLOATS


if __name__ == '__main__':
    # example: `python -m asr_eval.bench.check whisper-tiny`
    
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline', help='pipeline name')
    args = parser.parse_args()
    
    waveform: FLOATS = librosa.load('tests/testdata/formula1.mp3', sr=16_000)[0] # type: ignore
    
    pipeline_cls = get_pipeline(args.pipeline)
    assert issubclass(pipeline_cls, (TranscriberPipeline, TimedTranscriberPipeline))
    
    pipeline_obj = pipeline_cls()
    text = pipeline_obj.transcriber.transcribe(waveform)
    
    print('SUCCESS! TEXT:', text)