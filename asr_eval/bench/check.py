import argparse
from collections.abc import Sequence
from typing import cast
import re

import numpy as np

from asr_eval.bench.datasets import get_dataset, AudioSample
from asr_eval.bench.pipelines import get_pipeline
from asr_eval.streaming.caller import receive_transcription
from asr_eval.streaming.evaluation import make_sender
from asr_eval.streaming.model import StreamingASR, TranscriptionChunk
from asr_eval.utils.types import FLOATS


description = """A command line utility to check that a pipeline works."""

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    'pipeline',
    help='A pipeline name registered in asr_eval.',
)
parser.add_argument(
    '--audio',
    required=False,
    default='EN',
    help=(
        'Audio file path or pre-defined audios downloaded from Hugging Face:'
        ' EN (default) - 10 sec English audio'
        ', EN_LONG - 47 sec English audio'
        ', RU - 20 sec Russian audio'
        ', RU_LONG - 77 sec Russian audio.'
    ),
    metavar='PATH',
)
parser.add_argument(
    '--trim',
    required=False,
    type=float,
    help='If float, take the first N seconds of the audio.',
    metavar='N',
)


# --- Code for building docs ---
cli_block_for_docs = re.sub(
    r'usage: .*?.py',
    'usage: <strong>python -m asr_eval.bench.check</strong>',
    parser.format_help()
)
__doc__ = (
    # need a literal block (like .. code-block::) but with word wrap
    f'{description}\n\n.. raw:: html\n\n\t'
    + '<pre style="white-space: pre-wrap">'
    + cli_block_for_docs.replace('\n', '<br>')
    + '</pre>'
)
parser.description = description
# --- End code for building docs ---


def get_ru_dataset() -> Sequence[AudioSample]:
    # bond005/podlodka_speech
    return cast(Sequence[AudioSample], get_dataset('podlodka'))

def get_en_dataset() -> Sequence[AudioSample]:
    from datasets import load_dataset, Audio # type: ignore
    return cast(Sequence[AudioSample], (
        load_dataset('PolyAI/minds14', name='en-US', split='train')
        .cast_column('audio', Audio(sampling_rate=16_000)) # type: ignore
    ))

    
if __name__ == '__main__':
    args = parser.parse_args()
    
    match args.audio:
        case 'EN':
            dataset = get_en_dataset()
            waveform = dataset[0]['audio']['array']
        case 'EN_LONG':
            dataset = get_en_dataset()
            waveform = np.concatenate([
                # generating a long enough audio to test VAD
                dataset[0]['audio']['array'],
                np.zeros(16_000),
                dataset[1]['audio']['array'],
                np.zeros(16_000),
                dataset[2]['audio']['array'],
                np.zeros(16_000),
                dataset[3]['audio']['array'],
            ])
        case 'RU':
            dataset = get_ru_dataset()
            waveform = dataset[0]['audio']['array']
        case 'RU_LONG':
            dataset = get_ru_dataset()
            waveform = np.concatenate([
                # generating a long enough audio to test VAD
                dataset[0]['audio']['array'],
                np.zeros(16_000),
                dataset[1]['audio']['array'],
                np.zeros(16_000),
                dataset[2]['audio']['array'],
                np.zeros(16_000),
                dataset[3]['audio']['array'],
            ])
        case _:
            import librosa
            waveform: FLOATS = librosa.load(args.audio, sr=16_000)[0] # type: ignore
    
    if args.trim:
        waveform = waveform[:int(16_000 * args.trim)]
    
    pipeline_cls = get_pipeline(args.pipeline)
    pipeline_obj = pipeline_cls()
    if isinstance(pipeline_obj.transcriber, StreamingASR):
        cutoffs, sender = make_sender(waveform, pipeline_obj.transcriber)
        sender.start_sending(without_delays=True)
        output_chunks = list(receive_transcription(
            asr=pipeline_obj.transcriber, id=sender.id
        ))
        text = TranscriptionChunk.join(output_chunks)
        pipeline_obj.transcriber.stop_thread()
    else:
        text = pipeline_obj.transcriber.transcribe(waveform)
    print('SUCCESS! TEXT:', text)