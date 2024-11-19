import argparse
import json
import logging
import os

import datasets
import transformers
import valohai
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

import helpers

logger = logging.getLogger(__name__)


class FineTuner:
    def __init__(self, args):
        self.model_id = args.model_id
        self.train_data_path = args.train_data
        self.val_data_path = args.val_data
        self.output_dir = args.output_dir
        self.max_tokens = args.max_tokens
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.learning_rate = args.learning_rate
        self.do_eval = args.do_eval

        self.quantization_config = helpers.get_quantization_config()
        if self.quantization_config:
            self.optimizer = 'paged_adamw_8bit'
        else:
            self.optimizer = transformers.TrainingArguments.default_optim

        data_parallel_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        self.accelerator = Accelerator(fsdp_plugin=data_parallel_plugin)

        train_path = self.train_data_path or valohai.inputs('train_data').dir_path()
        val_path = self.val_data_path or valohai.inputs('val_data').dir_path()
        self.tokenized_train_dataset = datasets.load_from_disk(train_path)
        self.tokenized_eval_dataset = datasets.load_from_disk(val_path)

        self.tokenizer = helpers.get_tokenizer(self.model_id, self.max_tokens)
        self.model = helpers.get_model(self.model_id, self.quantization_config)
        self.model.gradient_checkpointing_enable()

        self.apply_peft()

    def apply_peft(self):
        model = prepare_model_for_kbit_training(self.model)
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
                'lm_head',
            ],
            bias='none',
            lora_dropout=0.05,
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, config)

        self.print_trainable_parameters()
        self.model = self.accelerator.prepare_model(model)

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}',
        )

    def train(self):
        checkpoints_output_dir = valohai.outputs().path(self.output_dir)
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_eval_dataset,
            args=transformers.TrainingArguments(
                output_dir=checkpoints_output_dir,
                warmup_steps=self.warmup_steps,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant': False},
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,
                logging_steps=1,
                bf16=False,
                tf32=False,
                optim=self.optimizer,
                logging_dir='./logs',
                save_strategy='steps',
                save_steps=10,
                eval_strategy='steps',
                eval_steps=10,
                do_eval=self.do_eval,
                report_to='none',
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[PrinterCallback],
        )

        self.model.config.use_cache = False
        trainer.train()

        model_output_dir = os.path.join(checkpoints_output_dir, 'best_model')
        trainer.save_model(model_output_dir)

        self.save_metadata(model_output_dir)

    @staticmethod
    def save_metadata(model_output_dir: str):
        project_name, exec_id = helpers.get_run_identification()

        metadata = {
            'valohai.dataset-versions': [
                {
                    'uri': f'dataset://mistral-models/{project_name}_{exec_id}',
                    'targeting_aliases': ['best_mistral_checkpoint'],
                    'valohai.tags': ['dev', 'mistral'],
                },
            ],
        }
        for file in os.listdir(model_output_dir):
            metadata_path = os.path.join(model_output_dir, f'{file}.metadata.json')
            with open(metadata_path, 'w') as outfile:
                json.dump(metadata, outfile)


class PrinterCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop('total_flos', None)
        print(json.dumps(logs))


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Fine-tune a model')
    # fmt: off
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-v0.1", help="Model identifier from Hugging Face")
    parser.add_argument("--train_data", type=str, help="Path to the training data")
    parser.add_argument("--val_data", type=str, help="Path to the validation data")
    parser.add_argument("--output_dir", type=str, default="finetuned_mistral", help="Output directory for checkpoints")
    parser.add_argument("--max_tokens", type=int, default=512, help="The maximum number of tokens that the model can process in a single forward pass")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--do_eval", action="store_true", help="Perform evaluation at the end of training")
    # fmt: on
    args = parser.parse_args()

    fine_tuner = FineTuner(args)
    fine_tuner.train()


if __name__ == '__main__':
    main()
