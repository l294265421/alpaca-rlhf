"""
Step1:
single gpu:
nohup sh run.sh --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step1_supervised_finetuning/main.py --sft_only_data_path MultiTurnAlpaca --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path decapoda-research/llama-7b-hf --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_seq_len 512 --learning_rate 3e-4 --gradient_accumulation_steps 32 --num_warmup_steps 100 --output_dir /root/autodl-tmp/rlhf/actor --lora_dim 8 --lora_module_name q_proj,k_proj --only_optimize_lora --deepspeed --zero_stage 0 > step1.log 2>&1 &

single node:
nohup sh run.sh --num_gpus 2 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step1_supervised_finetuning/main.py --sft_only_data_path MultiTurnAlpaca --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path decapoda-research/llama-7b-hf --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_seq_len 512 --learning_rate 3e-4 --gradient_accumulation_steps 32 --num_warmup_steps 100 --output_dir /root/autodl-tmp/rlhf/actor --lora_dim 8 --lora_module_name q_proj,k_proj --only_optimize_lora --deepspeed --zero_stage 0 > step1.log 2>&1 &


nohup sh run.sh --num_gpus 4 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step1_supervised_finetuning/main.py --sft_only_data_path MultiTurnAlpaca --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path decapoda-research/llama-7b-hf --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --max_seq_len 512 --learning_rate 3e-4 --num_train_epochs 1 --gradient_accumulation_steps 8 --num_warmup_steps 100 --output_dir /root/autodl-tmp/rlhf/actor --lora_dim 8 --lora_module_name q_proj,k_proj --only_optimize_lora --deepspeed --zero_stage 2 > step1.log 2>&1 &

step2:
single gpu:
nohup sh run.sh --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step2_reward_model_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path decapoda-research/llama-7b-hf --num_padding_at_beginning 0 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 32 --num_warmup_steps 100 --zero_stage 0 --deepspeed --output_dir /root/autodl-tmp/rlhf/critic --lora_dim 8 --lora_module_name q_proj,k_proj --only_optimize_lora > step2.log 2>&1 &

single node:
nohup sh run.sh --num_gpus 2 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step2_reward_model_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path decapoda-research/llama-7b-hf --num_padding_at_beginning 0 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 32 --num_warmup_steps 100 --zero_stage 0 --deepspeed --output_dir /root/autodl-tmp/rlhf/critic --lora_dim 8 --lora_module_name q_proj,k_proj --only_optimize_lora > step2.log 2>&1 &

Step3:
single gpu:
nohup sh run.sh --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step3_rlhf_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --actor_model_name_or_path /root/autodl-tmp/rlhf/actor/ --tokenizer_name_or_path decapoda-research/llama-7b-hf --critic_model_name_or_path /root/autodl-tmp/rlhf/critic --actor_zero_stage 0 --critic_zero_stage 0 --num_padding_at_beginning 0 --per_device_train_batch_size 4 --per_device_mini_train_batch_size 4 --gradient_accumulation_steps 32 --deepspeed --actor_lora_dim 4 --actor_lora_module_name q_proj,k_proj --critic_lora_dim 4 --critic_lora_module_name q_proj,k_proj --only_optimize_lora --output_dir /root/autodl-tmp/rlhf/final > step3.log 2>&1 &

single node:
nohup sh run.sh --num_gpus 2 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step3_rlhf_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --actor_model_name_or_path /root/autodl-tmp/rlhf/actor/ --tokenizer_name_or_path decapoda-research/llama-7b-hf --critic_model_name_or_path /root/autodl-tmp/rlhf/critic --actor_zero_stage 2 --critic_zero_stage 2 --num_padding_at_beginning 0 --per_device_train_batch_size 4 --per_device_mini_train_batch_size 4 --gradient_accumulation_steps 16 --deepspeed --actor_lora_dim 4 --actor_lora_module_name q_proj --critic_lora_dim 4 --critic_lora_module_name q_proj,k_proj --only_optimize_lora --output_dir /root/autodl-tmp/rlhf/final > step3.log 2>&1 &

"""
import os

from deepspeed.launcher.runner import main

os.environ["PATH"] = os.environ["PATH"] + ":/root/miniconda3/bin/"


if __name__ == '__main__':
    main()
