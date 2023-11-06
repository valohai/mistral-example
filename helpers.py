import json
import logging
import time

logger = logging.getLogger(__name__)


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
