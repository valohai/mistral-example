import json
import logging

import torch
from peft import PeftModel
from werkzeug.wrappers import Request, Response

from helpers import get_model, get_quantization_config, get_tokenizer, promptify

# Initialize logging
logger = logging.getLogger(__name__)

# Global variable for ModelInference
model_inference = None


class ModelInference:
    def __init__(self, model_id: str, checkpoint_path: str, max_tokens: int = 512) -> None:
        """Initialize model inference with the fine-tuned adapter model."""
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.max_tokens = max_tokens

        logger.info('Loading tokenizer...')
        self.tokenizer = get_tokenizer(self.model_id, self.max_tokens, add_eos_token=False)

        logger.info('Loading base model...')
        base_model = get_model(model_id=self.model_id, quantization_config=get_quantization_config())

        logger.info('Loading fine-tuned model...')
        self.ft_model = PeftModel.from_pretrained(base_model, self.checkpoint_path).eval()

    def get_meaning(self, sentence: str) -> str:
        """Generate a response and extract the meaning representation."""
        prompt = promptify(sentence=sentence)
        response = self.generate_response(prompt)

        try:
            meaning = response.split('### Meaning representation:')[1].split('\n')[1]
        except IndexError:
            raise ValueError(f'Failed to extract meaning from response: {response}')

        return meaning

    def generate_response(self, prompt: str) -> str:
        """Generate model response based on input prompt."""
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
        """Encode the prompt into tensor format."""
        return self.tokenizer(prompt, return_tensors='pt')

    def decode(self, model_outputs):
        """Decode model outputs into readable text."""
        return self.tokenizer.decode(model_outputs[0], skip_special_tokens=True).strip()


def load_model_if_needed():
    """Lazy-load the model only when needed."""
    global model_inference
    if model_inference is None:
        logger.info('Initializing ModelInference...')
        adapter_model_path = '.'
        model_inference = ModelInference(
            model_id='mistralai/Mistral-7B-v0.1',
            checkpoint_path=adapter_model_path,
            max_tokens=512,
        )


def predict(environ, start_response):
    """WSGI-compliant predict function to handle HTTP requests."""
    request = Request(environ)

    try:
        data = request.get_json()
        if not data or 'sentence' not in data:
            return Response(
                json.dumps({'error': "Missing 'sentence' in request"}),
                status=400,
                content_type='application/json',
            )(environ, start_response)

        sentence = data['sentence']
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), status=400, content_type='application/json')(
            environ,
            start_response,
        )

    # Load the model (only once)
    load_model_if_needed()

    # Generate response
    try:
        response_text = model_inference.get_meaning(sentence)
    except Exception as e:
        return Response(json.dumps({'error': str(e)}), status=500, content_type='application/json')(
            environ,
            start_response,
        )

    # Log Valohai metadata (optional tracking)
    print(json.dumps({'vh_metadata': {'sentence': sentence, 'response_length': len(response_text)}}))

    # Return JSON response
    response = Response(json.dumps({'sentence': sentence, 'response': response_text}), content_type='application/json')
    return response(environ, start_response)


# Run a local test server with `python predict.py`
if __name__ == '__main__':
    from werkzeug.serving import run_simple

    run_simple('0.0.0.0', 8000, predict)
