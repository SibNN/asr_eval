# type: ignore

import io
import argparse
from pathlib import Path
import datetime
import time
import shutil
from dataclasses import asdict
from contextlib import contextmanager

import datasets
from datasets import Audio, Dataset, load_dataset, IterableDataset
from pydub import AudioSegment
from tqdm.auto import tqdm
import librosa

from asr_eval.models.pyannote_wrapper import SpeakerDiarizationWrapper
from asr_eval.models.ast_wrapper import AudioSpectrogramTransformer


@contextmanager
def catchtime(name: str, disable: bool = False):
    """Prints elapsed time on exit from the context."""
    start = time.time()
    yield lambda: time.time() - start
    if not disable:
        print(f'{name}: {time.time() - start:.1f} seconds')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dataset', default='espnet/yodas')
    parser.add_argument('-n', '--input_name', required=False)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('--extract_audio', action='store_true')
    parser.add_argument('--bitrate', default='32k')
    parser.add_argument('--flush', action='store_true')
    parser.add_argument('--no_streaming', action='store_true')
    parser.add_argument('--n_shards', required=False)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ast', action='store_true')
    parser.add_argument('--ast_batch_size', default=256)
    parser.add_argument('--ast_min_length', default=1)
    parser.add_argument('--diarization', action='store_true')
    parser.add_argument('--diarization_segmentation_batch_size', default=256)
    parser.add_argument('--diarization_embedding_batch_size', default=128)
    parser.add_argument('--max_duration', required=False)

    args = parser.parse_args()

    if args.flush:
        # fixing a weird bug "No such file or directory ... 00000000.txt"
        # originating probably from the YODAS code
        shutil.rmtree(
            Path(datasets.config.HF_CACHE_HOME)
            / 'modules/datasets_modules/datasets/espnet--yodas',
            ignore_errors=True,
        )
        shutil.rmtree(
            Path(datasets.config.HF_CACHE_HOME)
            / 'modules/datasets_modules/datasets/espnet--yodas2',
            ignore_errors=True,
        )

    print(f'Loading {args.input_dataset} {args.input_name}')
    orig_dataset: IterableDataset = load_dataset(
        args.input_dataset,
        name=args.input_name,
        split='train',
        trust_remote_code=True,
        streaming=not bool(args.no_streaming)
    )
    print(f'Probing the loaded dataset')
    next(iter(orig_dataset))  # a test

    if args.ast:
        ast = AudioSpectrogramTransformer()
    
    if args.diarization:
        speaker_diarization = SpeakerDiarizationWrapper(
            segmentation_batch_size=args.diarization_segmentation_batch_size,
            embedding_batch_size=args.diarization_embedding_batch_size
        )

    n_shards = int(args.n_shards) or orig_dataset.num_shards
    for shard_idx in range(n_shards):

        filepath = Path(f'{args.output_dir}/{shard_idx:05d}-of-{n_shards:05d}')
        print(
            f'[{str(datetime.datetime.now())[:-7]}]'
            f' shard {shard_idx}/{n_shards}'
            f' {filepath}'
        )

        if filepath.exists():
            print('Already exists')
            continue
        
        start_time = time.time()

        source_dataset = (
            orig_dataset
            .shard(num_shards=n_shards, index=shard_idx)
            .cast_column('audio', Audio(decode=False))
        )

        collected_results = []
        for sample_idx, sample in enumerate(tqdm(source_dataset, disable=args.verbose)):
            if args.verbose:
                if 'video_id' in sample:
                    print(f'Sample {sample_idx} {sample["video_id"]}')
                else:
                    print(f'Sample {sample_idx}')
            sample = dict(**sample)  # otherwise it does not work for some reason
            orig_bytes = sample['audio']['bytes']
            # to mp3

            if args.max_duration is not None and sample['duration'] > float(args.max_duration):
                print(f'Skipping audio with duration {sample["duration"]}')
                continue
            
            if args.extract_audio:
                with catchtime('\tConverting to mp3 bytes', disable=not args.verbose):
                    audio = AudioSegment.from_file(io.BytesIO(orig_bytes))
                    audio.export(buffer := io.BytesIO(), format='mp3', bitrate=args.bitrate)
                    mp3_bytes = buffer.read()
                rel_audio_path = Path(f'audio/{sample["video_id"]}.mp3')
                audio_path = args.output_dir / rel_audio_path
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                with open(audio_path, 'wb') as f:
                    f.write(mp3_bytes)
                sample['audio'] = {'path': str(rel_audio_path)}

            if args.ast or args.diarization:
                with catchtime('\tLoading wav bytes', disable=not args.verbose):
                    # here we may already have AudioSegment, but there is no documented and consistent
                    # with librosa way to convert it into waveform array
                    # also, do we need to use sr=16_000 with AST?
                    waveform, sr = librosa.load(io.BytesIO(orig_bytes), sr=16_000)
                
                if args.verbose:
                    print(f'\tAudio length: {len(waveform) / sr:.1f} sec')

                if args.ast:
                    with catchtime('\tAST', disable=not args.verbose):
                        sample['ast'] = ast.predict_on_long_audio(
                            waveform,
                            batch_size=int(args.ast_batch_size),
                            sampling_rate=sr,
                            min_length=int(args.ast_min_length),
                        )
                
                if args.diarization:
                    with catchtime('\tDiarization', disable=not args.verbose):
                        segments, speaker_embeddings = speaker_diarization(waveform, sampling_rate=sr)
                        sample['segments'] = segments
                        sample['speaker_embeddings'] = speaker_embeddings
            
            collected_results.append(sample)

        with catchtime('Saving to disk', disable=not args.verbose):
            resulting_dataset = Dataset.from_list(collected_results)
            # resulting_dataset.to_parquet(tmp_path := filepath.with_stem('tmp'))
            resulting_dataset.save_to_disk(str(tmp_path := filepath.with_stem('tmp')))
            tmp_path.rename(filepath)  # to prevent truncated parquet files

        print(
            f'Elapsed {time.time() - start_time:.0f} sec'
            f', saved {filepath.stat().st_size / 1024 ** 2:.1f} MB'
        )