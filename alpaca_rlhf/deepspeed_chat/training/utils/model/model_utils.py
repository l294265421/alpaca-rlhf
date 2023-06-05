# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)

from transformers.deepspeed import HfDeepSpeedConfig
from deepspeed.runtime.zero import stage3
from .reward_model import RewardModel


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            # torch_dtype=torch.float16
        )

    # model.config.end_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = model.config.eos_token_id
    # model.config.bos_token_id = tokenizer.bos_token_id
    # model.config.end_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = tokenizer.pad_token_id

    #  利用Tensor Core优化GPU性能的几个小窍门 http://www.gpus.cn/gpus_list_page_techno_support_content?id=61
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    stage3.estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4,
                                                          num_nodes=1,
                                                          additional_buffer_factor=1.5)

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout)
    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)

    if rlhf_training:
        # critic model needs to load the weight here
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"
        critic_model.load_state_dict(
            torch.load(model_ckpt_path, map_location='cpu'))

    return critic_model
