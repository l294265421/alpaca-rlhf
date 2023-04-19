"""
Step1:
single_gpu: --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/training_model/step1_supervised_finetuning/main.py --gradient_accumulation_steps 2 --zero_stage 0 --deepspeed --output_dir /root/autodl-tmp/rlhf/actor

Step2:
single_gpu: --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/training_model/step2_reward_model_finetuning/main.py --num_padding_at_beginning 0 --gradient_accumulation_steps 2 --zero_stage 0 --lora_dim 8 --lora_module_name q_proj,k_proj --only_optimize_lora --deepspeed --output_dir /root/autodl-tmp/rlhf/critic

Step3: /tmp/pycharm_project_227/alpaca_rlhf/training_model/step3_rlhf_finetuning/main.py --actor_model_name_or_path /root/autodl-tmp/rlhf/actor --critic_model_name_or_path /root/autodl-tmp/rlhf/critic  --actor_zero_stage 0 --critic_zero_stage 0 --num_padding_at_beginning 0 --gradient_accumulation_steps 2 --deepspeed --enable_hybrid_engine --output_dir /root/autodl-tmp/rlhf/final
"""
import os

from deepspeed.launcher.runner import main

os.environ["PATH"] = os.environ["PATH"] + ":/root/miniconda3/bin/"


if __name__ == '__main__':
    main()
