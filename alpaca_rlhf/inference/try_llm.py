"""
nohup python chatbot_gradio.py > chatbot_gradio.log 2>&1 &
"""
import os
import sys
import random
import argparse

import gradio as gr
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import GenerationConfig
import mdtex2html


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='facebook/opt-350m',
                    choices=['decapoda-research/llama-7b-hf',
                             'facebook/opt-350m',
                             'gpt2'])
args = parser.parse_args()

path = args.path
model_name = path

if model_name == 'gpt2':
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id, device_map='auto')
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
                model_name,
                from_tf=bool(".ckpt" in model_name),
                config=model_config
            )

model.eval()


"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_new_tokens, num_beams, do_sample, temperature, top_p, history):
    # current_utterance = 'Human: {instruction} Assistant: '.format(instruction=input)
    current_utterance = '{instruction} '.format(instruction=input)
    if history:
        instruction = history + ' ' + current_utterance
    else:
        instruction = current_utterance
    prompt = instruction
    print('prompt: ' + prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    print('output: %s' % output)
    response = output[len(prompt):]

    chatbot.append((parse_text(input), parse_text(response)))
    history = instruction + response
    return chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], ''


with gr.Blocks() as demo:
    gr.HTML(f"""<h1 align="center">Alpaca-{model_name} (主要支持英文)</h1>""")
    gr.HTML("""<h1 align="center"><a href="https://github.com/l294265421/alpaca-rlhf">GitHub</a></h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=7).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_new_tokens = gr.Slider(0, 1024, value=128, step=1.0, label="Maximum Token Number", interactive=True)
            num_beams = gr.components.Slider(
                minimum=1, maximum=4, step=1, value=1, label="Beams"
            )
        with gr.Column(scale=1):
            do_sample = gr.components.Checkbox(value=False, label='Do sample')
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)

    history = gr.State('')
    # input, chatbot, max_new_tokens, num_beams, do_sample, temperature, top_p, history
    submitBtn.click(predict, [user_input, chatbot, max_new_tokens, num_beams, do_sample, temperature, top_p, history],
                    [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)


server_name = '0.0.0.0'
share_gradio = True
result = demo.queue().launch(server_name=server_name, share=share_gradio, prevent_thread_lock=True)
print('result:')
print(result[2], flush=True)
demo.block_thread()
