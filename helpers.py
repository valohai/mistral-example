from __future__ import annotations

import json
import logging
import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


def get_tokenizer(
    model_id: str,
    max_tokens: int,
    add_eos_token: bool = True,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=max_tokens,
        add_eos_token=add_eos_token,
    )
    if add_eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def get_model(model_id: str, quantization_config: BitsAndBytesConfig | None) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )


def promptify(sentence: str, meaning: str | None = None) -> str:
    prompt = f"""
        Given a target sentence construct the underlying meaning representation of the input sentence as a single 
        function with attributes and attribute values. This function should describe the target string accurately and 
        the function must be one of the following: 
            ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 
            'suggest', 'request_explanation', 'recommend', 'request_attribute'].
        The attributes must be one of the following: 
            ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 
            'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

        ### Target sentence:
        {sentence}
    
        ### Meaning representation:
        {meaning if meaning else ''}
    """
    return clean_prompt(prompt)


def clean_prompt(prompt: str) -> str:
    # force it into a single line to have a consistent baseline
    prompt = ' '.join(line.strip() for line in prompt.splitlines() if line.strip())

    # '###' should be preceded by a blank line
    prompt = prompt.replace('###', '\n\n###')

    # the first `:` on a line that starts with `###` should be followed by a newline
    lines = []
    for line in prompt.splitlines():
        if line.startswith('###') and ':' in line:
            line = ':\n'.join([subline.strip() for subline in line.split(':', 1)])
        lines.append(line.strip())

    # end prompt with a blank line
    if lines and lines[-1] != '':
        lines.append('')

    return '\n'.join(lines)


def get_run_identification():
    try:
        with open('/valohai/config/execution.json') as f:
            exec_details = json.load(f)
        project_name = exec_details['valohai.project-name'].split('/')[1]
        exec_id = exec_details['valohai.execution-id']
    except FileNotFoundError:
        project_name = 'test'
        exec_id = str(int(time.time()))
    return project_name, exec_id


def get_quantization_config():
    try:
        import bitsandbytes
        import torch
        from transformers import BitsAndBytesConfig

        assert bitsandbytes

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except Exception:
        logger.warning('Failed to initialize bitsandbytes config, not quantizing', exc_info=True)
        return None
