"""
Step1:
single_gpu:
nohup sh run.sh --num_gpus 4 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step1_supervised_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path /root/.cache/huggingface/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --num_warmup_steps 100 --output_dir /root/autodl-tmp/rlhf/actor --lora_dim 8 --lora_module_name q_proj,k_proj --only_optimize_lora --deepspeed --zero_stage 2 > step1.log 2>&1 &


Step2:
single_gpu:
nohup sh run.sh --num_gpus 4 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step2_reward_model_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path /root/.cache/huggingface/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348 --num_padding_at_beginning 0 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --num_warmup_steps 100 --zero_stage 2 --deepspeed --output_dir /root/autodl-tmp/rlhf/critic --lora_dim 4 --lora_module_name q_proj --only_optimize_lora > step2.log 2>&1 &

Step3:
single_gpu:
nohup sh run.sh /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step3_rlhf_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --actor_model_name_or_path /root/autodl-tmp/rlhf/actor/ --critic_model_name_or_path /root/autodl-tmp/rlhf/critic --actor_zero_stage 0 --critic_zero_stage 0 --num_padding_at_beginning 0 --per_device_train_batch_size 8 --per_device_mini_train_batch_size 8 --gradient_accumulation_steps 2 --deepspeed --enable_hybrid_engine --actor_gradient_checkpointing --output_dir /root/autodl-tmp/rlhf/final > step2.log 2>&1 &

"""
import os

from deepspeed.launcher.runner import main

os.environ["PATH"] = os.environ["PATH"] + ":/root/miniconda3/bin/"


if __name__ == '__main__':
    main()
