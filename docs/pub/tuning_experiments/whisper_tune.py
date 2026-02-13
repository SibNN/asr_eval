# python -m uv pip install datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio jupyter "torch<2.8" "torchaudio<2.8" "torchcodec==0.4"

# https://huggingface.co/blog/fine-tune-whisper
# https://discuss.huggingface.co/t/runtimeerror-backward-through-graph-with-whisper-medium-and-gradient-checkpointing-true/168571

from dataclasses import dataclass
from typing import Any, cast
import argparse

import torch
from datasets import load_dataset, Audio, Dataset # type: ignore
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='openai/whisper-medium')
parser.add_argument('--dataset', default='bond005/sova_rudevices')
parser.add_argument('--output-dir', default='tmp/fine-tuning-whisper')
args = parser.parse_args()


train_dataset = cast(Dataset, (
    load_dataset(args.dataset, split='train')
    .cast_column('audio', Audio(sampling_rate=16000)) # type: ignore
))


@dataclass
class WhisperDataCollator:
    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer
    decoder_start_token_id: int

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [self.preprocess_audio(sample) for sample in samples]
        batch = self.feature_extractor.pad(input_features, return_tensors='pt') # type: ignore

        # get the tokenized label sequences, pad the labels to max length
        label_features = [self.preprocess_transcription(sample) for sample in samples]
        labels_batch = self.tokenizer.pad(label_features, return_tensors='pt') # type: ignore

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100) # type: ignore

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item(): # type: ignore
            labels = labels[:, 1:] # type: ignore

        batch['labels'] = labels
        return batch # type: ignore

    def preprocess_audio(self, sample: dict[str, Any]) -> dict[str, Any]:
        waveform = sample['audio']['array']
        rate = sample['audio']['sampling_rate']
        features = self.feature_extractor(waveform, sampling_rate=rate).input_features[0] # type: ignore
        return {'input_features': features}

    def preprocess_transcription(self, sample: dict[str, Any]) -> dict[str, Any]:
        input_ids = self.tokenizer(sample['transcription']).input_ids # type: ignore
        return {'input_ids': input_ids}


model = WhisperForConditionalGeneration.from_pretrained(args.model) # type: ignore
model.generation_config.language = 'Russian' # type: ignore
model.generation_config.task = 'transcribe' # type: ignore
model.generation_config.forced_decoder_ids = None # type: ignore


data_collator = WhisperDataCollator(
    feature_extractor=WhisperFeatureExtractor.from_pretrained(args.model), # type: ignore
    tokenizer=WhisperTokenizer.from_pretrained( # type: ignore
        args.model,
        language=model.generation_config.language, # type: ignore
        task=model.generation_config.task, # type: ignore
    ),
    decoder_start_token_id=model.config.decoder_start_token_id, # type: ignore # type: ignore
)


training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    lr_scheduler_type='constant_with_warmup',
    warmup_steps=500,
    max_steps=50000,
    # gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={'use_reentrant': False},
    fp16=False,
    # fp16_full_eval=False,
    eval_strategy='no',
    save_strategy='steps',
    save_steps=100,
    save_only_model=True,
    logging_steps=100,
    load_best_model_at_end=False,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train() # type: ignore