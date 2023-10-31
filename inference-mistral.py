from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import os
from peft import PeftModel
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class ModelInference:
    def __init__(self, model_path, checkpoint_path, prompt):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512, padding_side="left",
                                                       add_eos_token=True)

        self.ft_model = self.load_checkpoint(model_path, checkpoint_path)

    def setup_accelerator(self):
        os.environ["WANDB_DISABLED"] = "true"
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        return Accelerator(fsdp_plugin=fsdp_plugin)

    def load_checkpoint(self, model_path, checkpoint_path):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, quantization_config=bnb_config)
        ft_model = PeftModel.from_pretrained(model, checkpoint_path)
        return ft_model.eval()

    def generate_response(self, prompt, max_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.ft_model.generate(**inputs, max_length=max_tokens, pad_token_id=2)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def main(args):
    model_path = '/valohai/inputs/model-base/'
    checkpoint_path = '/valohai/inputs/finetuned-checkpoint/'

    inference = ModelInference(model_path, checkpoint_path, args.prompt)
    response = inference.generate_response(args.prompt, args.max_tokens)
    print("Generated Response:")
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuned Model Inference")
    parser.add_argument("--prompt", type=str, help="Input prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum number of tokens in the generated response")

    args = parser.parse_args()
    main(args)
