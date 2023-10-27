from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from datasets import load_dataset
import os
import torch

import datasets

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# import valohai
#
# model_path = '/valohai/inputs/model/'
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,  # Mistral, same as before
#     device_map="auto",
#     trust_remote_code=True,
#     use_auth_token=True
# )
#
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token
#
# test_dataset = datasets.load_from_disk('/valohai/inputs/test_data/')
#
# print("Target Sentence: " + test_dataset[1]['target'])
# print("Meaning Representation: " + test_dataset[1]['meaning_representation'] + "\n")
#
# eval_prompt = """Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
# This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
# The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']
#
# ### Target sentence:
# Earlier, you stated that you didn't have strong feelings about PlayStation's Little Big Adventure. Is your opinion true for all games which don't have multiplayer?
#
# ### Meaning representation:
# """
#
# model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
#
# model.eval()
# with torch.no_grad():
#     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True))
#
#


import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelInference:
    def __init__(self, model_path, prompt):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512, padding_side="left",
                                                       add_eos_token=True)

    def generate_response(self, prompt, max_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_tokens, pad_token_id=2)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def main(args):
    model_path = '/valohai/inputs/model/'
    inference = ModelInference(model_path, args.prompt)
    response = inference.generate_response(args.prompt, args.max_tokens)
    print("Generated Response:")
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuned Model Inference")
    parser.add_argument("--prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of tokens in the generated response")

    args = parser.parse_args()
    main(args)
