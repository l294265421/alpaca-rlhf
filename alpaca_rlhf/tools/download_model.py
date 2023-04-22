from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)


model_name = 'facebook/opt-350m'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model_config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=model_config
        )
print('end')
