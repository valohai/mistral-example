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
        self.train_data_path = args.train_data
        self.val_data_path = args.val_data
        self.base_mistral_model = args.base_mistral_model or '/valohai/inputs/model/'
        self.output_dir = args.output_dir
        self.model_max_length = args.model_max_length
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.learning_rate = args.learning_rate
        self.do_eval = args.do_eval
        self.quantization_config = helpers.get_quantization_config()
        if self.quantization_config:
            self.optimizer = 'paged_adamw_8bit'
        else:
            self.optimizer = transformers.TrainingArguments.default_optim

        self.setup_accelerator()
        self.setup_datasets()
        self.setup_model()
        self.apply_peft()

    def setup_accelerator(self):
        os.environ['WANDB_DISABLED'] = 'true'
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        self.accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    def setup_datasets(self):
        train_path = self.train_data_path or valohai.inputs('train_data').dir_path()
        val_path = self.val_data_path or valohai.inputs('val_data').dir_path()

        self.tokenized_train_dataset = datasets.load_from_disk(train_path)
        self.tokenized_eval_dataset = datasets.load_from_disk(val_path)

    def setup_model(self):
        base_model_id = self.base_mistral_model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=self.quantization_config,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_id,
            model_max_length=self.model_max_length,
            padding_side='left',
            add_eos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.gradient_checkpointing_enable()

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}',
        )

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
            lora_dropout=0.05,  # Conventional
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, config)

        self.print_trainable_parameters()

        self.model = self.accelerator.prepare_model(model)

    def train(self):
        checkpoint_output_dir = valohai.outputs().path(self.output_dir)
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_eval_dataset,
            args=transformers.TrainingArguments(
                output_dir=checkpoint_output_dir,
                warmup_steps=self.warmup_steps,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=self.max_steps,
                learning_rate=self.learning_rate,  # Want about 10x smaller than the Mistral learning rate
                logging_steps=10,
                bf16=False,
                tf32=False,
                optim=self.optimizer,
                logging_dir='./logs',  # Directory for storing logs
                save_strategy='steps',  # Save the model checkpoint every logging step
                save_steps=10,  # Save checkpoints every 10 steps
                eval_strategy='steps',  # Evaluate the model every logging step
                eval_steps=17,  # Evaluate and save checkpoints every 17 steps
                do_eval=self.do_eval,  # Perform evaluation at the end of training
                report_to=None,
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[PrinterCallback],
        )

        self.model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()
        model_save_dir = os.path.join(checkpoint_output_dir, 'best_model')

        trainer.save_model(model_save_dir)

        # save metadata
        self.save_valohai_metadata(model_save_dir)

    @staticmethod
    def save_valohai_metadata(save_dir):
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
        for file in os.listdir(save_dir):
            md_path = os.path.join(save_dir, f'{file}.metadata.json')
            metadata_path = valohai.outputs().path(md_path)
            with open(metadata_path, 'w') as outfile:
                json.dump(metadata, outfile)


class PrinterCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop('total_flos', None)
        print(json.dumps(logs))


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Fine-tune a model')

    # Add arguments based on your script's needs
    # fmt: off
    parser.add_argument("--base_mistral_model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base mistral from hugging face")
    parser.add_argument("--train_data", type=str, help="Path to the training data")
    parser.add_argument("--val_data", type=str, help="Path to the validation data")
    parser.add_argument("--output_dir", type=str, default="finetuned_mistral", help="Output directory for checkpoints")
    parser.add_argument("--model_max_length", type=int, default=512, help="Maximum length for the model")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=2.5e-5, help="Learning rate")
    parser.add_argument("--do_eval", action="store_true", help="Perform evaluation at the end of training")
    # fmt: on

    args = parser.parse_args()

    fine_tuner = FineTuner(args)
    fine_tuner.train()


if __name__ == '__main__':
    main()
