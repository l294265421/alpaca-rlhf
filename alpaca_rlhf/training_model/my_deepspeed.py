"""
Step1:
single_gpu: --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/training_model/step1_supervised_finetuning/main.py --gradient_accumulation_steps 2 --lora_dim 128 --zero_stage 0 --deepspeed --output_dir /root/autodl-tmp/actor

Step2:
single_gpu: --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/training_model/step2_reward_model_finetuning/main.py --num_padding_at_beginning 0 --gradient_accumulation_steps 2 --zero_stage 0 --deepspeed --output_dir /root/autodl-tmp/critic

Step3: /tmp/pycharm_project_227/alpaca_rlhf/training_model/step3_rlhf_finetuning/main.py --actor_model_name_or_path /root/autodl-tmp/actor --critic_model_name_or_path /root/autodl-tmp/critic  --actor_zero_stage 0 --critic_zero_stage 0 --num_padding_at_beginning 0 --gradient_accumulation_steps 2 --deepspeed --actor_lora_dim 128 --enable_hybrid_engine --actor_gradient_checkpointing --output_dir /root/autodl-tmp/final
"""
import os

from deepspeed.launcher.runner import main

os.environ["PATH"] = os.environ["PATH"] + ":/root/miniconda3/bin/"


if __name__ == '__main__':
    main()
