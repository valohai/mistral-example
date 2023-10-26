# Import necessary libraries
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    TextStreamer,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

class MistralFineTuning:
    def __init__(self, base_model, dataset_name, new_model):
        self.base_model = base_model
        self.dataset_name = dataset_name
        self.new_model = new_model
        self.model = None
        self.tokenizer = None

    def train(self):
        dataset = self.load_dataset()
        self.load_base_model()
        self.load_tokenizer()
        self.add_adapters()
        # self.monitor_llm()
        self.set_hyperparameters()
        self.fine_tune_model(dataset)
        user_prompt = "what is Newton's 2nd law and its formula"
        self.generate_response(user_prompt)
        self.clear_memory()
        self.reload_base_model()
        self.reload_tokenizer()

    def load_dataset(self):
        # Importing the dataset
        dataset = load_dataset(self.dataset_name, split="train")
        return dataset

    def load_base_model(self):
        # Load base model (Mistral 7B)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map={"": 0},
        )

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.model.gradient_checkpointing_enable()

    def load_tokenizer(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_eos_token = True
        self.tokenizer.add_bos_token, self.tokenizer.add_eos_token

    def add_adapters(self):
        # Adding the adapters in the layers
        self.model = prepare_model_for_kbit_training(self.model)
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        )
        self.model = get_peft_model(self.model, self.peft_config)

    def monitor_llm(self):
        # Monitor the LLM
        wandb.login(key="Wandb authorization key")
        run = wandb.init(project='Fine tuning mistral 7B', job_type="training", anonymous="allow")

    def set_hyperparameters(self):
        # Set hyperparameters
        self.training_arguments = TrainingArguments(
            output_dir="./results",
            num_train_epochs=2,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            optim="paged_adamw_8bit",
            save_steps=1000,
            logging_steps=30,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.3,
            group_by_length=True,
            lr_scheduler_type="constant",
        )

    def fine_tune_model(self, dataset):
        # Fine-tune the model
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=self.peft_config,
            max_seq_length=None,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            packing=False,
        )

        # Save the fine-tuned model
        trainer.model.save_pretrained(self.new_model)

        self.model.config.use_cache = True
        self.model.eval()

    def generate_response(self, user_prompt):
        # Generate a response
        runtimeFlag = "cuda:0"
        system_prompt = 'The conversation between Human and AI assisatance named Gathnex\n'
        B_INST, E_INST = "[INST]", "[/INST]"

        prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n{E_INST}"

        inputs = self.tokenizer([prompt], return_tensors="pt").to(runtimeFlag)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        _ = self.model.generate(**inputs, streamer=streamer, max_new_tokens=200)

    def clear_memory(self):
        # Clear the memory footprint
        del self.model
        torch.cuda.empty_cache()

    def reload_base_model(self):
        # Reload the base model
        base_model_reload = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )
        self.model = PeftModel.from_pretrained(base_model_reload, self.new_model)
        self.model = self.model.merge_and_unload()

    def reload_tokenizer(self):
        # Reload tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"


if __name__ == "__main__":
    base_model = "mistralai/Mistral-7B-v0.1"
    dataset_name = "gathnex/Gath_baize"
    new_model = "gathnex/Gath_mistral_7b"

    mistral_fine_tuning = MistralFineTuning(base_model, dataset_name, new_model)
    mistral_fine_tuning.train()
