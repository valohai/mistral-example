from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from datasets import load_dataset
import os
import valohai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import transformers
import argparse
from datetime import datetime


class TrainMistral:

    def __init__(self):
        os.environ["WANDB_DISABLED"] = "true"

        self.fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )

        self.accelerator = Accelerator(fsdp_plugin=self.fsdp_plugin)
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

        self.model = None
        self.tokenizer = None

    def load_datasets(self):
        self.train_dataset = load_dataset('gem/viggo', split='train')
        self.eval_dataset = load_dataset('gem/viggo', split='validation')
        self.test_dataset = load_dataset('gem/viggo', split='test')

    def setup_model(self, model_path, bnb_config):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config,
                                                          local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=512,
            padding_side="left",
            add_eos_token=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, prompt):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
        This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
        The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

        ### Target sentence:
        {data_point["target"]}

        ### Meaning representation:
        {data_point["meaning_representation"]}
        """
        return self.tokenize(full_prompt)

    def prepare_datasets(self):
        tokenized_train_dataset = self.train_dataset.map(self.generate_and_tokenize_prompt)
        tokenized_val_dataset = self.eval_dataset.map(self.generate_and_tokenize_prompt)
        return tokenized_train_dataset, tokenized_val_dataset

    def print_example(self, tokenized_train_dataset):
        print(tokenized_train_dataset[4]['input_ids'])
        print(len(tokenized_train_dataset[4]['input_ids']))

    def base_model_example(self, test_dataset):
        print("Target Sentence: " + test_dataset[1]['target'])
        print("Meaning Representation: " + test_dataset[1]['meaning_representation'] + "\n")

        eval_prompt = """Your evaluation prompt here..."""
        model_input = self.tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        self.model.eval()
        with torch.no_grad():
            print(self.tokenizer.decode(self.model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0],
                                        skip_special_tokens=True))


    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def apply_peft(self, lora_config):
        self.model = get_peft_model(self.model, lora_config)
        self.print_trainable_parameters()

        self.model = self.accelerator.prepare_model(self.model)

    def train_model(self, tokenized_train_dataset, tokenized_val_dataset, model_path):
        if torch.cuda.device_count() > 1:
            self.model.is_parallelizable = True
            self.model.model_parallel = True

        project = "viggo-finetune"
        base_model_name = "mistral"
        run_name = base_model_name + "-" + project
        output_dir = valohai.outputs().path(run_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                warmup_steps=5,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                max_steps=50,
                learning_rate=2.5e-5,
                logging_steps=50,
                bf16=False,
                tf32=False,
                optim="paged_adamw_8bit",
                logging_dir="./logs",
                save_strategy="steps",
                save_steps=50,
                evaluation_strategy="steps",
                eval_steps=50,
                do_eval=True,
                report_to=None
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        self.model.config.use_cache = False
        trainer.train()

    def load_base_model(self, base_model_id, bnb_config):
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token


def main(args):
    train_mistal = TrainMistral()

    train_mistal.load_datasets()
    train_mistal.setup_model(args.model_path, args.bnb_config)

    tokenized_train_dataset, tokenized_val_dataset = train_mistal.prepare_datasets()

    train_mistal.print_example(tokenized_train_dataset)

    train_mistal.base_model_example(train_mistal.test_dataset)

    if args.peft:
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        train_mistal.apply_peft(lora_config)

    train_mistal.train_model(tokenized_train_dataset, tokenized_val_dataset, args.model_path)

    train_mistal.load_base_model(args.base_model_id, args.bnb_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/valohai/inputs/model/")
    parser.add_argument("--bnb_config", type=str, default=None)
    parser.add_argument("--base_model_id", type=str, default="/valohai/inputs/model/")

    args = parser.parse_args()
    main(args)
