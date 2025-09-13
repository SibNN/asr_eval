from collections.abc import Iterable
from tqdm.auto import tqdm
from typing import cast
from pathlib import Path
import pickle

from asr_eval.bench.datasets import get_dataset
from asr_eval.bench.datasets import AudioSample
from asr_eval.bench.recording import Recording
from asr_eval.streaming.evaluation import default_evaluation_pipeline

# from asr_eval.models.vosk_streaming_wrapper import VoskStreaming
# model_save_name = 'vosk-model-ru-0.42-0.5s'
# model = VoskStreaming(model_name='vosk-model-ru-0.42', chunk_length_sec=0.5)

# from asr_eval_extras.sibnn_psz_wrapper import SibnnPszStreamingWrapper
# model_save_name = 'psz-dev-branch-commit-hash-be884dcf'
# model = SibnnPszStreamingWrapper(src_path='tmp/psz/src', clarification=False)

# from asr_eval_extras.sibnn_psz_wrapper import SibnnPszStreamingWrapper
# model_save_name = 'psz-dev-branch-commit-hash-be884dcf-clarification'
# model = SibnnPszStreamingWrapper(src_path='tmp/psz/src', clarification=True)

# from asr_eval.models.whisper_wrapper import WhisperLongformWrapper
# from asr_eval.streaming.wrappers import OfflineToStreaming
# model_save_name = 'whisper-large-v3-quasi-streaming-0.5sec'
# model = OfflineToStreaming(WhisperLongformWrapper('openai/whisper-large-v3'), interval=0.5)

# from asr_eval.models.gigaam_wrapper import GigaAMShortformCTC
# from asr_eval.models.base.longform import LongformCTC
# from asr_eval.streaming.wrappers import OfflineToStreaming
# model_save_name = 'gigaam-ctc-quasi-streaming-0.5sec'
# model = OfflineToStreaming(LongformCTC(GigaAMShortformCTC()), interval=0.5)

from asr_eval.models.t_one_wrapper import TOneStreaming
model_save_name = 't-one'
model = TOneStreaming()

model.start_thread()

dataset_name = 'common-voice-17.0'

dataset = get_dataset(dataset_name)

max_samples = 500
if len(dataset) > max_samples:
    dataset = dataset.take(max_samples)

for sample_idx, sample in enumerate(tqdm(cast(Iterable[AudioSample], dataset))):
    input_path = Path(f'tmp/timings/{dataset_name}/{sample_idx}.pkl')
    output_path = Path(f'tmp/streaming_evals/{model_save_name}/{dataset_name}/{sample_idx}.pkl')
    if not input_path.exists():
        continue
    if output_path.exists():
        print(f'Already exists: {output_path}')
        continue
    timed_transcription = pickle.loads(input_path.read_bytes())
    recording = Recording(
        transcription=timed_transcription,
        waveform=sample['audio']['array'],
    )
    eval = default_evaluation_pipeline(
        recording, model, partial_alignment_interval=0.5
    )
    del eval.sender.history_lock
    del eval.sender._thread # type: ignore
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_bytes(pickle.dumps(eval))

model.stop_thread()