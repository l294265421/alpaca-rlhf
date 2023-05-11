#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
import socket
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)
from transformers import LlamaTokenizer
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import wandb

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from alpaca_rlhf.deepspeed_chat.training.utils.model.model_utils import create_critic_model
from alpaca_rlhf.deepspeed_chat.training.utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from alpaca_rlhf.deepspeed_chat.training.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from alpaca_rlhf.deepspeed_chat.training.utils.ds_utils import get_train_ds_config
from alpaca_rlhf.deepspeed_chat.training.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from alpaca_rlhf.deepspeed_chat.training.utils import datetime_utils

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='4,4,2',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args

def print_args(model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for p in model.parameters():
        n_params = torch.prod(torch.tensor(p.shape)).item()
        if p.requires_grad:
            n_trainable_params += n_params
        else:
            n_nontrainable_params += n_params
    print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))


def main():
    args = parse_args()

    wandb.login(key="6c52cf61837ddd8efa62755b86139c41bbad09ec")
    project_name = 'rlhf'
    experiment_name = 'rlhf-step2'
    run_dir = os.path.join(args.data_output_path, 'log', project_name, experiment_name)
    if not os.path.exists(run_dir):
        os.makedirs(str(run_dir))
    wandb.init(config=args,
               entity='knowl',
               project=project_name,
               name=experiment_name + '_' + datetime_utils.now(),
               dir=run_dir,
               job_type="training",
               reinit=True,
               notes=socket.gethostname(),
               )

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    # ds_config['wandb'] = {"enabled": True, "group": "finetune777", "team": "finetune777", "project": "deepspeed-chat-2"}

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,
    #                                            fast_tokenizer=False)
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=False)
    # tokenizer.pad_token = tokenizer.eos_token
    if 'llama' in args.model_name_or_path.lower():
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.add_eos_token = True

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   disable_dropout=args.disable_dropout)

    print_args(rm_model)
    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)
    print_args(rm_model)


    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        rejected_scores = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_scores"].sum().float()
            rejected_scores += outputs["rejected_mean_scores"].sum().float()
            # if step == 99:  # For faster evaluation and debugging
            #     break
        acc = correct_predictions / total_predictions
        scores = scores / total_predictions
        rejected_scores = rejected_scores / total_predictions
        try:
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
            rejected_scores = get_all_reduce_mean(rejected_scores).item()
        except:
            pass
        return scores, rejected_scores, acc

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, rejected_scores, acc = evaluation_reward(rm_model, eval_dataloader)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, "
        f"reject_last_scores (higher is better) : {rejected_scores},  "
        f"acc (higher is better) : {acc}",
        args.global_rank)
    if args.global_rank == 0:
        wandb.log({
            'Eval/epoch': -1,
            'Eval/acc': reward_score,
            'Eval/acc': rejected_scores,
            'Eval/acc': acc,
        })

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]

            correct_predictions = (chosen > rejected).sum() * 1.0 / chosen.shape[0]
            reward = outputs["chosen_mean_scores"].mean().float()
            r_reward = outputs["rejected_mean_scores"].mean().float()
            print_rank_0(f'step: {step} loss:{loss}, '
                         f'correct_predictions: {correct_predictions}, '
                         f'reward: {reward} '
                         f'r_reward: {r_reward} ',
                         args.global_rank)

            if args.global_rank == 0:
                wandb.log({
                    'Train/epoch': epoch,
                    'Train/step': step,
                    'Train/loss': loss,
                    'Train/reward': reward,
                    'Train/r_reward': r_reward,
                    'Train/lr': lr_scheduler.get_lr(),
                    'Train/reward_diff': reward - r_reward
                })

        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)
        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        reward_score, rejected_scores, acc = evaluation_reward(rm_model, eval_dataloader)
        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, "
            f"reject_last_scores (higher is better) : {rejected_scores},  "
            f"acc (higher is better) : {acc}",
            args.global_rank)
        # chosen_last_scores (higher is better) : -0.37704116106033325, reject_last_scores (higher is better) : -0.41206246614456177,  acc (higher is better) : 0.564919114112854
        rm_model.tput_timer.update_epoch_count()

        if args.global_rank == 0:
            wandb.log({
                'Eval/epoch': epoch,
                'Eval/acc': reward_score,
                'Eval/acc': rejected_scores,
                'Eval/acc': acc,
            })
    wandb.finish()
    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)

        if args.global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(rm_model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
