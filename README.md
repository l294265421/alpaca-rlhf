# alpaca-rlhf
Finetuning alpaca with RLHF (Reinforcement Learning with Human Feedback). The base model is from [my-alpaca](https://github.com/l294265421/my-alpaca) and [multi-turn-alpaca](https://github.com/l294265421/multi-turn-alpaca), which train LLaMA with the original alpaca dataset and a multi-turn dialogue dataset respectively.

## Tips
- DeepSpeed-Chat implements PPO following [2017-Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) except for the entropy bonus.


## Stey by Step
- [Bootstrap Script](alpaca_rlhf/my_deepspeed.py)
    - step1: --num_gpus 2 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step1_supervised_finetuning/main.py --sft_only_data_path MultiTurnAlpaca --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path decapoda-research/llama-7b-hf --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --max_seq_len 512 --learning_rate 3e-4 --gradient_accumulation_steps 32 --num_warmup_steps 100 --output_dir /root/autodl-tmp/rlhf/actor --lora_dim 8 --lora_module_name q_proj,k_proj --only_optimize_lora --deepspeed --zero_stage 0
    - step2: --num_gpus 2 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step2_reward_model_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --model_name_or_path decapoda-research/llama-7b-hf --num_padding_at_beginning 0 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --gradient_accumulation_steps 32 --num_warmup_steps 100 --zero_stage 0 --deepspeed --output_dir /root/autodl-tmp/rlhf/critic --lora_dim 8 --lora_module_name q_proj,k_proj --only_optimize_lora
    - step3: --num_gpus 2 /tmp/pycharm_project_227/alpaca_rlhf/deepspeed_chat/training/step3_rlhf_finetuning/main.py --data_output_path /root/autodl-tmp/rlhf/tmp/ --actor_model_name_or_path /root/autodl-tmp/rlhf/actor/ --tokenizer_name_or_path decapoda-research/llama-7b-hf --critic_model_name_or_path /root/autodl-tmp/rlhf/critic --actor_zero_stage 2 --critic_zero_stage 2 --num_padding_at_beginning 0 --per_device_train_batch_size 4 --per_device_mini_train_batch_size 4 --gradient_accumulation_steps 16 --deepspeed --actor_lora_dim 4 --actor_lora_module_name q_proj --critic_lora_dim 4 --critic_lora_module_name q_proj,k_proj --only_optimize_lora --output_dir /root/autodl-tmp/rlhf/final

## References

### Articles
- [如何正确复现 Instruct GPT / RLHF?](https://zhuanlan.zhihu.com/p/622134699)

### Sources
- [Awesome RLHF](https://github.com/opendilab/awesome-RLHF)

### Tools
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- [trlx](https://github.com/CarperAI/trlx)

### Datasets
- [Stanford Human Preferences Dataset (SHP)](https://huggingface.co/datasets/stanfordnlp/SHP)
- [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
  - [hh-rlhf](https://github.com/anthropics/hh-rlhf)
    - Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback [[paper](https://arxiv.org/abs/2204.05862)]
    - [Dahoas/static-hh](https://huggingface.co/datasets/Dahoas/static-hh)
    - [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static)
- GPT-4-LLM
  - [GitHub](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
  - [Paper](https://arxiv.org/pdf/2304.03277.pdf)
  - [Site](https://instruction-tuning-with-gpt-4.github.io/)
- Open-Assistant
  - [Site](https://open-assistant.io/zh)
  - [GitHub](https://github.com/LAION-AI/Open-Assistant)
  - [Paper](./papers/2023-OpenAssistant%20Conversations%20-%20Democratizing%20Large%20Language%20Model%20Alignment.pdf)

### Repositories
- [my-alpaca](https://github.com/l294265421/my-alpaca)
- [multi-turn-alpaca](https://github.com/l294265421/multi-turn-alpaca)
