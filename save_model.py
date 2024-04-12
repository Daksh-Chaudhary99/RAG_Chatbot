import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from Constants import (hf_auth, model_id, model_path)

def does_directory_exist(directory):
    '''Check if a directory exists'''
    return os.path.exists(directory)


def save_model():
    '''Downloads and saves the LLM'''

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token = hf_auth)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16, use_auth_token = hf_auth)
    
    os.makedirs(model_path, exist_ok=True)

    model.save_pretrained(model_path, from_pt=True) 
    tokenizer.save_pretrained(model_path, from_pt=True)


if __name__ == "__main__":
    save_model()
