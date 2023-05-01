import argparse
from types import SimpleNamespace

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import set_seed
from transformers import TrainingArguments
from transformers import AutoConfig, GPT2LMHeadModel
import evaluate
import wandb
import numpy as np

import params
from customtrainer import CustomTrainer


CONTEXT_LENGTH = 512


# Commented parameters correspond to the small model
default_config = SimpleNamespace(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=60,
    logging_steps=60,
    logging_first_step=True,
    save_total_limit=5,
    save_steps=60,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    warmup_ratio=0.01,
    weight_decay=0.01,
    seed=1,
    load_best_model_at_end=True,
    report_to="wandb",
    prediction_loss_only=False,
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description="Process hyper-parameters")
    argparser.add_argument(
        "--output_dir",
        type=str,
        default=default_config.output_dir,
        help="Output directory",
    )
    argparser.add_argument(
        "--num_train_epochs",
        type=int,
        default=default_config.num_train_epochs,
        help="num train epochs",
    )
    argparser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=default_config.per_device_train_batch_size,
        help="per device train batch size",
    )
    argparser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=default_config.per_device_eval_batch_size,
        help="per device eval batch size",
    )
    argparser.add_argument(
        "--evaluation_strategy",
        type=str,
        default=default_config.evaluation_strategy,
        help="evaluation strategy",
    )
    argparser.add_argument(
        "--save_strategy",
        type=str,
        default=default_config.save_strategy,
        help="save strategy",
    )
    argparser.add_argument(
        "--eval_steps", type=int, default=default_config.eval_steps, help="eval steps"
    )
    argparser.add_argument(
        "--logging_steps",
        type=int,
        default=default_config.logging_steps,
        help="logging steps",
    )
    argparser.add_argument(
        "--logging_first_step",
        type=bool,
        default=default_config.logging_first_step,
        help="logging first step",
    )
    argparser.add_argument(
        "--save_total_limit",
        type=int,
        default=default_config.save_total_limit,
        help="save total limit",
    )
    argparser.add_argument(
        "--save_steps", type=int, default=default_config.save_steps, help="save steps"
    )
    argparser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=default_config.lr_scheduler_type,
        help="lr scheduler type",
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=default_config.learning_rate,
        help="learning rate",
    )
    argparser.add_argument(
        "--warmup_ratio",
        type=float,
        default=default_config.warmup_ratio,
        help="warmup ratio",
    )
    argparser.add_argument(
        "--weight_decay",
        type=float,
        default=default_config.weight_decay,
        help="weight decay",
    )
    argparser.add_argument("--seed", type=int, default=default_config.seed, help="seed")
    argparser.add_argument(
        "--load_best_model_at_end",
        type=float,
        default=default_config.load_best_model_at_end,
        help="load best model at end",
    )
    argparser.add_argument(
        "--report_to", type=str, default=default_config.report_to, help="report to"
    )
    argparser.add_argument(
        "--prediction_loss_only",
        type=bool,
        default=default_config.prediction_loss_only,
        help="prediction loss only",
    )
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


def get_raw_data_and_tokenizer():
    raw_datasets = load_dataset("TristanBehrens/js-fakes-4bars")
    # Change for respective tokenizer
    tokenizer = AutoTokenizer.from_pretrained("juancopi81/js-fakes-4bars_test")
    return raw_datasets, tokenizer


def tokenize(element, tokenizer):
    # Replace this based on Dataset
    context_length = CONTEXT_LENGTH
    outputs = tokenizer(
        element["text"],
        truncation=True,  # Removing element longer that context size, no effect in JSB
        max_length=context_length,
        padding=False,
    )
    return {"input_ids": outputs["input_ids"]}


def create_tokenized_dataset(raw_datasets, tokenizer):
    # Create tokenize dataset
    tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    return tokenized_datasets


def create_model(tokenizer):
    # Change this based on size of the data
    n_layer = 4
    n_head = 2
    n_emb = 512

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_positions=CONTEXT_LENGTH,
        n_layer=n_layer,
        n_head=n_head,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        n_embd=n_emb,
    )

    model = GPT2LMHeadModel(config)
    return model


def compute_metrics_fn(eval_pred):
    metrics = dict()
    accuracy_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics.update(accuracy_metric.compute(references=labels, predictions=predictions))
    return metrics


def train(train_config):
    config = vars(train_config)
    set_seed(config["seed"])
    raw_datasets, tokenizer = get_raw_data_and_tokenizer()
    tokenized_datasets = create_tokenized_dataset(
        raw_datasets=raw_datasets, tokenizer=tokenizer
    )
    model = create_model(tokenizer)
    run = wandb.init(project=params.WANDB_PROJECT, job_type="training", config=config)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_args = TrainingArguments(**config)
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    parse_args()
    train(default_config)
