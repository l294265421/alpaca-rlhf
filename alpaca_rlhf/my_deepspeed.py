"""
Step1:
single_gpu: --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step1_supervised_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path facebook/opt-350m --gradient_accumulation_steps 2 --num_warmup_steps 100 --output_dir /root/autodl-tmp/rlhf/actor --deepspeed

Step2:
single_gpu: --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step2_reward_model_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path facebook/opt-350m --num_padding_at_beginning 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 2 --zero_stage 0 --deepspeed --output_dir /root/autodl-tmp/rlhf/critic

Step3:
single_gpu: /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step3_rlhf_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --actor_model_name_or_path facebook/opt-350m --critic_model_name_or_path /root/autodl-tmp/rlhf/critic --actor_zero_stage 0 --critic_zero_stage 0 --num_padding_at_beginning 1 --per_device_train_batch_size 8 --per_device_mini_train_batch_size 8 --gradient_accumulation_steps 2 --deepspeed --enable_hybrid_engine --actor_gradient_checkpointing --output_dir /root/autodl-tmp/rlhf/final

"""
import os

from deepspeed.launcher.runner import main

os.environ["PATH"] = os.environ["PATH"] + ":/root/miniconda3/bin/"


if __name__ == '__main__':
    main()
