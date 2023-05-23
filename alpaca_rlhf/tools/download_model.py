from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig

from transformers import GPT2TokenizerFast

from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer


model_name = 'gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model_config = AutoConfig.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
            model_name,
            from_tf=bool(".ckpt" in model_name),
            config=model_config
        )
print('end')
