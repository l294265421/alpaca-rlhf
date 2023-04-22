from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b-chinese",
                                          trust_remote_code=True, cache_dir='/root/autodl-tmp/models/')
model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b-chinese", trust_remote_code=True, cache_dir='/root/autodl-tmp/models/')
model = model.half().cuda()

inputs = tokenizer("凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。", return_tensors="pt")
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
inputs = {key: value.cuda() for key, value in inputs.items()}
outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
print(tokenizer.decode(outputs[0].tolist()))
print('end')
