# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
#
# model_path = "/Users/daol/PycharmProjects/Synatra-1.3B/synatra"  # 또는 다른 모델 경로
#
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token
#
# model = AutoModelForCausalLM.from_pretrained(model_path).to("mps" if torch.backends.mps.is_available() else "cpu")
#
# prompt = "User: 된장찌개 레시피 알려줘\nAssistant:"
#
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#
# output = model.generate(
#     **inputs,
#     max_new_tokens=1000,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.9,
#     pad_token_id=tokenizer.pad_token_id,
# )
#
# generated_text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
#
# print(" 응답:", generated_text)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/Users/daol/PycharmProjects/Synatra-1.3B/synatra"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)



def generate_text(prompt: str) -> str:
    formatted_prompt = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response
