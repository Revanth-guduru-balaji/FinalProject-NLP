import logging
import math
import os
import argparse
from functools import partial
from tqdm.auto import tqdm
import random
import wandb
import nltk
import datasets
import numpy as np
from packaging import version

import transformers
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_metric
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

#import packages for pytorch tpu
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp



# Setup logging
logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


transformers.utils.logging.set_verbosity_warning()

def parse_args():
    """This function creates argument parser and parses the scrip input arguments."""
    parser = argparse.ArgumentParser(description="Train machine translation transformer model")

    # Required arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default = "output_dir",
        help=("Where to store the final model. "
              "Should contain the source and target tokenizers in the following format: "
              r"output_dir/{source_lang}_tokenizer and output_dir/{target_lang}_tokenizer. "
              "Both of these should be directories containing tokenizer.json files."
        ),
    )
   
    # Data arguments
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="en-de",
        help=("Many datasets in Huggingface Dataset repository have multiple versions or configs. "
              "For the case of machine translation these usually indicate the language pair like "
              "en-es or zh-fr or similar. To look up possible configs of a dataset, "
              "find it on huggingface.co/datasets."),
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Whether to use a small subset of the dataset for debugging.",
    )
    # Model arguments
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="The maximum total sequence length for source and target texts after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=None,
        help="Overwrite the cached training and evaluation sets",
    )

    # Training arguments
    parser.add_argument(
        "--device",
        default=None,                                                 #"cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu) on which the code should run",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout rate of the Transformer encoder",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=5000,
        help="Perform evaluation every n network updates.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Compute and log training batch metrics every n steps.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=transformers.SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--generation_type",
        choices=["greedy", "beam_search"],
        default="beam_search",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help=("Beam size for beam search generation. "
              "Decreasing this parameter will make evaluation much faster, "
              "increasing this (until a certain value) would likely improve your results."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--wandb_project", 
        default="pre-trained-bart",
        help="wandb project name to log metrics to"
    )
    parser.add_argument(
        "--model_name", 
        default="facebook/bart-base",
        help="To define the model name"
    )
    parser.add_argument(
        "--dataset_name", 
        default="cnn_dailymail",
        help="To define the dataset name"
    )
    parser.add_argument(
        "--dataset_version", 
        default="3.0.0",
        help="To define the dataset version"
    )
    parser.add_argument(
        "--metric",
        default="rouge",
        help="To define the metric"

    )

    args = parser.parse_args()


    return args






encoder_max_length = 256
decoder_max_length = 64

nltk.download("punkt", quiet=True)
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def preprocess_function(examples,tokenizer,max_input_length,max_target_length):
    inputs = [ex for ex in examples['article']]
    targets = [ex for ex in examples['highlights']]
    source_tokenized = tokenizer(inputs, max_length=max_input_length, truncation=True)
    target_tokenized = tokenizer(targets, max_length=max_target_length, truncation=True)
    batch = {k: v for k, v in source_tokenized.items()}
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch


def evaluate_model(
    model,
    dataloader,
    *,
    tokenizer,
    device,
    max_seq_length,
    generation_type,
    beam_size,
    metric
):
    n_generated_tokens = 0
    model.eval()
    for batch in tqdm(dataloader, desc="Evaluation"):
        with torch.inference_mode():
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            key_padding_mask = batch["encoder_padding_mask"].to(device)
            generated_tokens = model.generate(
                input_ids,
                key_padding_mask=key_padding_mask,
                max_length=max_seq_length,
                kind=generation_type,
                beam_size=beam_size,
            )
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            for pred in decoded_preds:
                n_generated_tokens += len(tokenizer(pred)["input_ids"])

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    model.train()
    eval_metric = metric.compute()
    evaluation_results = {
        "rouge1": eval_metric["rouge1"].mid.fmeasure * 100,
        "rouge2":eval_metric["rouge2"].mid.fmeasure * 100,
        "rougeL":eval_metric["rougeL"].mid.fmeasure * 100,
        "rougeLsum":eval_metric["rougeLsum"].mid.fmeasure * 100,
        "generation_length": n_generated_tokens / len(dataloader.dataset),
    }
    return evaluation_results, input_ids, decoded_preds, decoded_labels












args = parse_args()
wandb.init(project='pre-trained-bart',config=args)
###############################################################################
# Part 1: Load the data
###############################################################################

# Make sure output directory exists, if not create it
os.makedirs(args.output_dir, exist_ok=True)

# Load the datasets
dataset = load_dataset("cnn_dailymail", '3.0.0')
metric = load_metric(args.metric)
###############################################################################
# Part 2: Create the model and load the tokenizers
###############################################################################
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
device = xm.xla_device()
model = model.to(device)
###############################################################################
# Part 3: Pre-process the data
###############################################################################
# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = dataset["train"].column_names
preprocess_function_wrapped = partial(
    preprocess_function,
    max_input_length=encoder_max_length,
    max_target_length=decoder_max_length,
    tokenizer=tokenizer
)
processed_datasets = dataset.map(
    preprocess_function_wrapped,
    batched=True,
    num_proc=8,
    remove_columns=column_names,
    load_from_cache_file=not None,
    desc="Running tokenizer on dataset",
)
train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["validation"] if "validaion" in processed_datasets else processed_datasets["test"]


def main(index):
    ###############################################################################
    # Part 4: Create PyTorch dataloaders that handle data shuffling and batching
    ###############################################################################
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding='max_length')

    collation_function_for_seq2seq_wrapped = partial(
        data_collator
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,sampler=train_sampler,  collate_fn=collation_function_for_seq2seq_wrapped, batch_size=args.batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler,collate_fn=collation_function_for_seq2seq_wrapped, batch_size=args.batch_size
    )

    ###############################################################################
    # Part 5: Create optimizer and scheduler
    ###############################################################################
    learning_rate = 2e-5
    batch_size = 16
    num_train_epochs = 4
    weight_decay = 0.01

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate * xm.xrt_world_size(),
        weight_decay=weight_decay,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = transformers.get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps))

    # Log a pre-processed training example to make sure the pre-processing does not have bugs in it
    # and we do not input garbage to our model.
    batch = next(iter(train_dataloader))
    logger.info("Look at the data that we input into the model, check that it looks like what we expect.")
    for index in random.sample(range(len(batch)), 2):
        logger.info(f"Decoded input_ids: {tokenizer.decode(batch['input_ids'][index])}")
        logger.info("\n")

    ###############################################################################
    # Part 6: Training loop
    ###############################################################################
    global_step = 0
    para_loader_train = pl.ParallelLoader(train_dataloader, [device])
    para_loader_eval = pl.ParallelLoader(eval_dataloader, [device])
    # iterate over epochs
    for epoch in range(num_train_epochs):
        model.train()  # make sure that model is in training mode, e.g. dropout is enabled

        # iterate over batches
        for batch in para_loader_train.per_device_loader(device):
            input_ids = batch["input_ids"].to(device)
            decoder_input_ids = batch["decoder_input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            optimizer.zero_grad()
            loss = logits.loss()

            loss.backward()
            xm.optimizer_step(optimizer)
            lr_scheduler.step()
            

            progress_bar.update(1)
            global_step += 1

            wandb.log(
                {
                    "train_loss": loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            
            if global_step % args.eval_every_steps == 0 or global_step == args.max_train_steps:
                eval_results, last_input_ids, last_decoded_preds, last_decoded_labels = evaluate_model(
                    model=model,
                    dataloader=para_loader_eval.per_device_loader(device),
                    tokenizer=tokenizer,
                    device=device,
                    max_seq_length=decoder_max_length,
                    generation_type=args.generation_type,
                    beam_size=args.beam_size,
                    metric=metric
                )
                # YOUR CODE ENDS HERE
                wandb.log(
                    {
                        "eval/rouge1": eval_results["rouge1"].mid.fmeasure,
                        "eval/rouge2": eval_results["rouge2"].mid.fmeasure,
                        "eval/rougeL": eval_results["rougeL"].mid.fmeasure,
                        "eval/rougeLsum": eval_results["rougeLsum"].mid.fmeasure,
                        "eval/generation_length": eval_results["generation_length"],
                    },
                    step=global_step,
                )
                logger.info("Generation example:")
                random_index = random.randint(0, len(last_input_ids) - 1)
                logger.info(f"Input sentence: {tokenizer.decode(last_input_ids[random_index], skip_special_tokens=True)}")
                logger.info(f"Generated sentence: {last_decoded_preds[random_index]}")
                logger.info(f"Reference sentence: {last_decoded_labels[random_index][0]}")

                logger.info("Saving model checkpoint to %s", args.output_dir)
                model.save_pretrained(args.output_dir,save_function=xm.save)

            if global_step >= args.max_train_steps:
                break

    ###############################################################################
    # Part 8: Save the model
    ###############################################################################

    logger.info("Saving final model checkpoint to %s", args.output_dir)
    model.save_pretrained(args.output_dir,save_function=xm.save)

    logger.info("Uploading tokenizer, model and config to wandb")
    wandb.save(os.path.join(args.output_dir, "*"))

    logger.info(f"Script finished succesfully, model saved in {args.output_dir}")


if __name__ == "__main__":
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")
    xmp.spawn(main,nprocs=8)