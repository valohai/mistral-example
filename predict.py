import json
import logging

import torch
from peft import PeftModel
from werkzeug.wrappers import Request, Response

from helpers import get_model, get_quantization_config, get_tokenizer, promptify

# Initialize logging
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None


def load_finetuned_model(adapter_config_path, adapter_model_path, model_id='mistralai/Mistral-7B-v0.1', max_tokens=512):
    """Lazily loads the fine-tuned model and tokenizer using adapter configuration."""
    global model, tokenizer

    if model is None or tokenizer is None:
        logger.info('Loading tokenizer...')
        tokenizer = get_tokenizer(model_id, max_tokens, add_eos_token=False)

        logger.info('Loading base model...')
        base_model = get_model(model_id=model_id, quantization_config=get_quantization_config())

        logger.info('Loading fine-tuned model...')
        model = PeftModel.from_pretrained(base_model, adapter_model_path, config=adapter_config_path).eval()


def predict(environ, start_response):
    """WSGI-compliant predict function to handle HTTP requests."""
    # Parse request
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

    # Load model (if not already loaded)
    adapter_config_path = 'finetuned_mistral/best_model/adapter_config.json'
    adapter_model_path = 'finetuned_mistral/best_model/adapter_model.bin'
    load_finetuned_model(adapter_config_path, adapter_model_path)

    # Generate response
    prompt = promptify(sentence=sentence)
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        logger.info(f'Generating response with max {512} tokens...')
        outputs = model.generate(
            **inputs,
            max_length=512,
            pad_token_id=tokenizer.pad_token_id,
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Log Valohai metadata (optional tracking)
    print(json.dumps({'vh_metadata': {'sentence': sentence, 'response_length': len(response_text)}}))

    # Return JSON response
    response = Response(json.dumps({'sentence': sentence, 'response': response_text}), content_type='application/json')
    return response(environ, start_response)


# Run a local test server with `python predict.py`
if __name__ == '__main__':
    from werkzeug.serving import run_simple

    run_simple('0.0.0.0', 8000, predict)
