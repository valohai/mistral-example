import argparse
import valohai
import json
from datasets import load_dataset
from transformers import AutoTokenizer


class DataPreprocessor:
    def __init__(self):
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def load_datasets(self):
        self.train_dataset = load_dataset('gem/viggo', split='train')
        self.eval_dataset = load_dataset('gem/viggo', split='validation')
        self.test_dataset = load_dataset('gem/viggo', split='test')

    def prepare_datasets(self, tokenizer, generate_and_tokenize_prompt):
        tokenized_train_dataset = self.train_dataset.map(generate_and_tokenize_prompt)
        tokenized_val_dataset = self.eval_dataset.map(generate_and_tokenize_prompt)
        return tokenized_train_dataset, tokenized_val_dataset

    def generate_and_tokenize_prompt(self, data_point, tokenizer):
        full_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
        This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
        The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

        ### Target sentence:
        {data_point["target"]}

        ### Meaning representation:
        {data_point["meaning_representation"]}
        """
        return tokenizer(full_prompt, truncation=True, max_length=512, padding="max_length")

    def load_and_prepare_data(self):
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', model_max_length=512,
                                                  padding_side="left", add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_train_dataset, tokenized_val_dataset = self.prepare_datasets(tokenizer, lambda
            data_point: self.generate_and_tokenize_prompt(data_point, tokenizer))
        return tokenized_train_dataset, tokenized_val_dataset, self.test_dataset

    def save_dataset(self, dataset, tag='train'):
        f = open('/valohai/config/execution.json')
        exec_details = json.load(f)
        project_name = exec_details['valohai.project-name'].split('/')[1]
        exec_id = exec_details['valohai.execution-id']

        metadata = {
            "valohai.dataset-versions": [{
                'uri': f"dataset://viggo/{project_name}_{tag}_{exec_id}",
                'targeting_aliases': ['dev'],
                "valohai.tags": ["dev", "mistral"],
            }]
        }

        save_path = valohai.outputs().path(f'encoded_{tag}')
        dataset.save_to_disk(save_path)

        metadata_path = valohai.outputs().path(f'encoded_{tag}.metadata.json')
        with open(metadata_path, 'w') as outfile:
            json.dump(metadata, outfile)


if __name__ == "__main__":
    data_preprocessor = DataPreprocessor()
    data_preprocessor.load_datasets()

    tokenized_train_dataset, tokenized_val_dataset, test_dataset = data_preprocessor.load_and_prepare_data()

    data_preprocessor.save_dataset(tokenized_train_dataset, 'train')
    data_preprocessor.save_dataset(tokenized_val_dataset, 'val')
    data_preprocessor.save_dataset(test_dataset, 'test')
