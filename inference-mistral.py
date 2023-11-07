import argparse
import logging

import torch
import transformers
from peft import PeftModel

from helpers import get_quantization_config

logger = logging.getLogger(__name__)


class ModelInference:
    def __init__(self, model_path: str, checkpoint_path: str) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=512,
            padding_side='left',
            add_eos_token=True,
        )
        logger.info('Loading model...')
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=get_quantization_config(),
        )
        logger.info('Creating PEFT model...')
        self.ft_model = PeftModel.from_pretrained(model, checkpoint_path).eval()

    def generate_response(self, prompt: str, max_tokens: int = 50) -> str:
        inputs = self.prepare_prompt(prompt)
        with torch.no_grad():
            logger.info('Generating up to %d tokens...', max_tokens)
            outputs = self.ft_model.generate(**inputs, max_length=max_tokens, pad_token_id=2)
        logger.info('Decoding...')
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def prepare_prompt(self, prompt):
        test_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
        This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
        The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

        ### Target sentence:
            {prompt}
        ### Meaning representation:
        """
        return self.tokenizer(test_prompt, return_tensors='pt')


def run(args):
    inference = ModelInference(
        model_path=args.base_mistral_model,
        checkpoint_path=args.checkpoint_path,
    )
    response = inference.generate_response(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    print('Generated Response:')
    print(response)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Fine-tuned Model Inference')
    # fmt: off
    parser.add_argument('--base_mistral_model', type=str, default='mistralai/Mistral-7B-v0.1', help='Base mistral from hugging face')
    parser.add_argument('--checkpoint_path', default='/valohai/inputs/finetuned-checkpoint/')
    parser.add_argument('--max_tokens', type=int, default=305, help='Maximum number of tokens in response')
    parser.add_argument('--model_path', default='/valohai/inputs/model-base/')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for text generation')
    # fmt: on

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
