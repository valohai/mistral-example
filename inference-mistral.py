from __future__ import annotations

import argparse
import logging

import torch
import transformers
import valohai
from peft import PeftModel

from helpers import get_quantization_config

logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self, base_mistral_model: str, checkpoint_path: str | None) -> None:
        self.checkpoint_path = checkpoint_path or valohai.inputs('finetuned-checkpoint').dir_path()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_mistral_model,
            model_max_length=512,
            padding_side='left',
            add_eos_token=True,
        )

        logger.info('Loading model...')
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_mistral_model,
            quantization_config=get_quantization_config(),
        )

        logger.info('Creating PEFT model...')
        self.ft_model = PeftModel.from_pretrained(model, self.checkpoint_path).eval()

    def generate_response(self, prompt: str, max_tokens: int = 50) -> str:
        inputs = self.encode(prompt)

        with torch.no_grad():
            logger.info('Generating up to %d tokens...', max_tokens)
            outputs = self.ft_model.generate(**inputs, max_length=max_tokens, pad_token_id=2)

        return self.decode(outputs)

    def encode(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt')

    def decode(self, model_outputs):
        text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        return text.strip()


def run(args):
    prompt = valohai.parameters('prompt', args.prompt).value
    if not prompt:
        raise ValueError('--prompt argument is required when running outside of Valohai')

    inference = ModelInference(
        base_mistral_model=args.base_mistral_model,
        checkpoint_path=args.checkpoint_path,
    )
    response = inference.generate_response(
        prompt=prompt,
        max_tokens=args.max_tokens,
    )
    print('Generated Response:')
    print(response)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Fine-tuned Model Inference')
    # fmt: off
    parser.add_argument('--base_mistral_model', type=str, default='mistralai/Mistral-7B-v0.1', help='Mistral model path or id from Hugging Face')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--max_tokens', type=int, default=305, help='Maximum number of tokens in response')
    parser.add_argument('--prompt', type=str, help='Input prompt for text generation')
    # fmt: on
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
