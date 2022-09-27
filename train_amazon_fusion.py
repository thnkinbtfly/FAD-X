from datasets import load_dataset, concatenate_datasets
from transformers.adapters.composition import Stack, Fuse
from transformers import AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterConfig, TrainingArguments, AdapterTrainer, EvalPrediction
import numpy as np
import os
import argparse
from transformers import set_seed



import os

from transformers import MultiLingAdapterArguments
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AdapterArgs(MultiLingAdapterArguments):
    train_adapter: bool = field(default=False, metadata={"help": "Train an adapter instead of the full model."})
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration. Either an identifier or a path to a file."}
    )
    adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the adapter configuration."}
    )
    adapter_reduction_factor: Optional[float] = field(
        default=None, metadata={"help": "Override the reduction factor of the adapter configuration."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language adapter configuration. Either an identifier or a path to a file."}
    )
    lang_adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the language adapter configuration."}
    )
    lang_adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the language adapter configuration."}
    )
    load_adapter: List[str] = field(
        default_factory=lambda: [], metadata={"help": "Pre-trained adapter module to be loaded from Hub."}
    )
    language: List[str] = field(default_factory=lambda: [], metadata={"help": "The training language, e.g. 'en' for English."})

    load_lang_adapter: List[str] = field(
        default_factory=lambda: [], metadata={"help": "Pre-trained language adapter module to be loaded from Hub."}
    )
    fusion_of_stacks: bool = field(default=True)
    with_head: bool = field(default=True)

    output_dir: str = field(default="")
    lang: str = field(default="")
    num_samples: int = field(default=200000)
    epochs: int = field(default=8)

from transformers import HfArgumentParser

def main():
    parser = HfArgumentParser(AdapterArgs)
    args, = parser.parse_args_into_dataclasses()
    assert isinstance(args, AdapterArgs)

    task = 'amazon_reviews_multi'
    lang = args.lang
    dataset_en = load_dataset(task, lang, cache_dir=os.path.expanduser('~/dataset/cache'))

    set_seed(42)
    if args.num_samples != 200000:
        dataset_en['train'] = dataset_en['train'].select(np.random.choice(200000, args.num_samples, replace=False))
        args.output_dir += "/{}".format(args.num_samples)

    print("Task: {}, lang: {}".format(task, lang))
    print(dataset_en.num_rows)
    print(dataset_en['train'].features)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def encode_batch(examples):
        """Encodes a batch of input data using the model tokenizer."""
        all_encoded = {"input_ids": [], "attention_mask": [], 'stars': []}
        # Iterate through all examples in this batch

        for review, label in zip(examples['review_body'], examples['stars']):
            encoded = tokenizer(review, max_length=60, truncation=True, padding='max_length')

            all_encoded["input_ids"].append(encoded["input_ids"])
            all_encoded["attention_mask"].append(encoded["attention_mask"])
            all_encoded['stars'].append(label - 1)
        return all_encoded

    def preprocess_dataset(dataset):
        # Encode the input data
        dataset = dataset.map(encode_batch, batched=True)
        # The transformers model expects the target class column to be named "labels"
        dataset = dataset.rename_column("stars", "labels")
        # Transform to pytorch tensors and only output the required columns
        dataset.set_format(columns=["input_ids", "attention_mask", "labels"])
        return dataset

    dataset_en = preprocess_dataset(dataset_en)

    config = AutoConfig.from_pretrained(
        "xlm-roberta-base",
    )
    model = AutoAdapterModel.from_pretrained(
        "xlm-roberta-base",
        config=config,
    )

    stacked_adapters = []
    task_adapters = []
    # load a pre-trained from Hub if specified

    def normalize_name(s):
        s = s.replace("@", "")
        s = s.replace("/", "")
        s = s.replace("checkpoint-5000", "")
        s = s.replace("samples200000", "200K")
        return s
    # Load the language adapters
    lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
    if args.fusion_of_stacks:
        for task_adapter, lang_adapter in zip(args.load_adapter, args.load_lang_adapter):
            print(task_adapter, lang_adapter)
            task_adapter_name = model.load_adapter(
                task_adapter,
                load_as=normalize_name(task_adapter),
                with_head=args.with_head
            )
            # optionally load a pre-trained language adapter
            # resolve the language adapter config
            # load the language adapter from Hub

            # if language is mhr, it seems critical to use relu-pfeiffer
            lang_adapter_name = model.load_adapter(
                lang_adapter,
                config=lang_adapter_config,
                load_as=normalize_name(lang_adapter),
            )
            print(task_adapter_name, lang_adapter_name)
            stacked_adapters.append(Stack(lang_adapter_name, task_adapter_name))
            task_adapters.append(task_adapter_name)

        fuse_adapter_setup = Fuse(*stacked_adapters) # note that the invertible adapter is used for only the first adapter
        model.add_adapter_fusion(fuse_adapter_setup)
        # if not args.with_head:
        model.add_classification_head('fusefusefuse', num_labels=5)
        # Freeze all model weights except of those of this adapter
        model.set_active_adapters(fuse_adapter_setup)
        model.train_adapter_fusion(fuse_adapter_setup) # fusion의 weight만 사용함

    else:
        assert len(args.load_lang_adapter) == 1
        for task_adapter in args.load_adapter:
            task_adapter_name = model.load_adapter(
                task_adapter,
                load_as=normalize_name(task_adapter),
                with_head=args.with_head
            )
            task_adapters.append(task_adapter_name)

        lang_adapter = args.load_lang_adapter[0]
        # if language is mhr, it seems critical to use relu-pfeiffer
        lang_adapter_name = model.load_adapter(
            lang_adapter,
            config=lang_adapter_config,
            load_as=normalize_name(lang_adapter),
        )
        fuse_adapter_setup = Fuse(*task_adapters) # note that the invertible adapter is used for only the first adapter
        model.add_adapter_fusion(fuse_adapter_setup)
        # if not args.with_head:
        model.add_classification_head('fusefusefuse', num_labels=5)
        # Freeze all model weights except of those of this adapter
        # model.set_active_adapters(fuse_adapter_setup)
        model.train_adapter_fusion(fuse_adapter_setup) # fusion의 weight만 사용함
        model.set_active_adapters(Stack(lang_adapter_name, fuse_adapter_setup))

    print("Active adapters: ", model.active_adapters)

    # # Unfreeze and activate stack setup
    # model.active_adapters = Stack(lang, task_adapter)

    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_total_limit=2,
        logging_steps=100,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        evaluation_strategy='epoch',
        save_strategy='epoch',
    )

    # train_dataset = concatenate_datasets([dataset_en["train"], dataset_en["validation"]])

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_en['train'],
        eval_dataset=dataset_en['validation'],
        compute_metrics=compute_accuracy,

    )
    trainer.train()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    predictions, labels, metrics = trainer.predict(dataset_en['test'], metric_key_prefix="predict")

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

if __name__ == '__main__':
    main()