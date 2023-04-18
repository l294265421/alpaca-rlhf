# alpaca-rlhf
Finetuning alpaca with RLHF (Reinforcement Learning with Human Feedback). The base model is from [my-alpaca](https://github.com/l294265421/my-alpaca) and [multi-turn-alpaca](https://github.com/l294265421/multi-turn-alpaca), which train LLaMA with the original alpaca dataset and a multi-turn dialogue dataset respectively.

## Stey by Step
### Remote Debug Using PyCharm
- [Bootstrap Script](alpaca_rlhf/training_model/my_deepspeed.py)
  - single gpu args
    - step1: --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/training_model/step1_supervised_finetuning/main.py --gradient_accumulation_steps 2 --lora_dim 128 --zero_stage 0 --deepspeed --output_dir /root/autodl-tmp/actor
    - step2: --num_gpus 1 /tmp/pycharm_project_227/alpaca_rlhf/training_model/step2_reward_model_finetuning/main.py --num_padding_at_beginning 0 --gradient_accumulation_steps 2 --zero_stage 0 --deepspeed --output_dir /root/autodl-tmp/critic
    - step3: /tmp/pycharm_project_227/alpaca_rlhf/training_model/step3_rlhf_finetuning/main.py --actor_model_name_or_path /root/autodl-tmp/actor --critic_model_name_or_path /root/autodl-tmp/critic  --actor_zero_stage 0 --critic_zero_stage 0 --num_padding_at_beginning 0 --gradient_accumulation_steps 2 --deepspeed --actor_lora_dim 128 --enable_hybrid_engine --actor_gradient_checkpointing --output_dir /root/autodl-tmp/final

### Prepare Data

### Training Reward Model

### Finetuning Language Model

## References

### Sources
- [Awesome RLHF](https://github.com/opendilab/awesome-RLHF)

### Tools
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

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

### Tips
- DeepSpeed-Chat implements PPO following [2017-Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) except for the entropy bonus.
