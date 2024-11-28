from __future__ import annotations

import argparse
import logging

import torch
import valohai
from peft import PeftModel

from helpers import get_model, get_quantization_config, get_tokenizer, promptify

logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(
        self,
        model_id: str,
        checkpoint_path: str | None,
        max_tokens: int,
    ) -> None:
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path or valohai.inputs('finetuned-checkpoint').dir_path()
        self.max_tokens = max_tokens

        logger.info('Loading tokenizer...')
        self.tokenizer = get_tokenizer(self.model_id, self.max_tokens, add_eos_token=False)

        logger.info('Loading model...')
        model = get_model(model_id=self.model_id, quantization_config=get_quantization_config())

        logger.info('Creating PEFT model...')
        self.ft_model = PeftModel.from_pretrained(model, self.checkpoint_path).eval()

    def get_meaning(self, sentence: str) -> str:
        prompt = promptify(sentence=sentence)
        response = self.generate_response(prompt)

        try:
            meaning = response.split('### Meaning representation:')[1].split('\n')[1]
        except IndexError:
            raise ValueError(f'Failed to extract meaning from response: {response}')

        return meaning

    def generate_response(self, prompt: str) -> str:
        inputs = self.encode(prompt)
        inputs = inputs.to(self.ft_model.device)

        with torch.no_grad():
            logger.info(f'Generating up to {self.max_tokens} tokens...')
            outputs = self.ft_model.generate(
                **inputs,
                max_length=self.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.decode(outputs)

    def encode(self, prompt):
        return self.tokenizer(prompt, return_tensors='pt')

    def decode(self, model_outputs):
        text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
        return text.strip()


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Fine-tuned Model Inference')
    # fmt: off
    parser.add_argument("--model_id", type=str, default='mistralai/Mistral-7B-v0.1', help="Model identifier from Hugging Face")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens in response")
    parser.add_argument("--sentence", type=str, required=True, help="The sentence to analyze")
    # fmt: on
    args = parser.parse_args()

    inference = ModelInference(
        model_id=args.model_id,
        checkpoint_path=args.checkpoint_path,
        max_tokens=args.max_tokens,
    )
    meaning = inference.get_meaning(sentence=args.sentence)
    print(meaning)


if __name__ == '__main__':
    main()
