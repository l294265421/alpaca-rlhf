from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer

base_model: str = "decapoda-research/llama-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(base_model)

model = LlamaForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
)
print('end')
