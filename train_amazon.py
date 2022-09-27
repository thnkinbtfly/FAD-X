from datasets import load_dataset, concatenate_datasets
from transformers.adapters.composition import Stack
from transformers import AutoTokenizer, AutoConfig, AutoAdapterModel, AdapterConfig, TrainingArguments, AdapterTrainer, EvalPrediction
import numpy as np
import os
import argparse


import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./training_output", help="output directory")
    parser.add_argument("--lang", type=str, default='en', choices=['en', 'zh', 'de', 'ja', 'es', 'fr'])
    parser.add_argument("--num_samples", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--no_train", default=False, action='store_true')
    parser.add_argument("--load_adapter", type=str, default="", help="task adapter you want to load")

    args = parser.parse_args()

    return args

def main():

    args = get_args()

    from transformers import set_seed
    set_seed(42)
    task = 'amazon_reviews_multi'
    lang = args.lang
    dataset_en = load_dataset(task, lang, cache_dir=os.path.expanduser('~/dataset/cache'))

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

    # Load the language adapters
    lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
    if lang == 'en':
        model.load_adapter("en/wiki@ukp", config=lang_adapter_config)
    if lang == 'zh':
        model.load_adapter("zh/wiki@ukp", config=lang_adapter_config)
    if lang == 'de':
        model.load_adapter("de/wiki@ukp", config=lang_adapter_config)
    if lang == 'es':
        model.load_adapter("es/wiki@ukp", config=lang_adapter_config)
    if lang == 'fr':
        model.load_adapter("fr/wiki@ukp", config=lang_adapter_config)
    if lang == 'ja':
        model.load_adapter("ja/wiki@ukp", config=lang_adapter_config)

    # Add a new task adapter
    task_adapter = "{}_{}".format(task, lang)
    if args.load_adapter:
        model.load_adapter(
            args.load_adapter,
            load_as=task_adapter,
            with_head=False,
        )
    else:
        model.add_adapter(task_adapter)

    # Add a classification head for our target task
    model.add_classification_head(task_adapter, num_labels=5)

    model.train_adapter([task_adapter])

    # Unfreeze and activate stack setup
    model.active_adapters = Stack(lang, task_adapter)

    args.output_dir += "/{}".format(lang)
    training_args = TrainingArguments(
        learning_rate=1e-5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=100,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        save_total_limit=3,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # eval_steps=500,
        # save_steps=500,
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

    if not args.no_train:
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
        # trainer.evaluate()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    predictions, labels, metrics = trainer.predict(dataset_en['test'], metric_key_prefix="predict")

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

if __name__ == '__main__':
    main()