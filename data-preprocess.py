import argparse
import json
import logging
import os

import valohai
from datasets import load_dataset
from transformers import AutoTokenizer

from helpers import get_run_identification


class DataPreprocessor:
    def __init__(self, args):
        self.data_path = args.data_path
        self.model_max_length = args.model_max_length
        self.tokenizer = args.tokenizer
        self.train_dataset = load_dataset('csv', data_files=valohai.inputs('dataset').path('train.csv'))
        self.eval_dataset = load_dataset('csv', data_files=valohai.inputs('dataset').path('validation.csv'))
        self.test_dataset = load_dataset('csv', data_files=valohai.inputs('dataset').path('test.csv'))

    def prepare_datasets(self, generate_and_tokenize_prompt):
        tknzd_train_dataset = self.train_dataset.map(generate_and_tokenize_prompt)
        tknzd_val_dataset = self.eval_dataset.map(generate_and_tokenize_prompt)
        return tknzd_train_dataset, tknzd_val_dataset

    def generate_and_tokenize_prompt(self, data_point, tokenizer):
        full_prompt = f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
        This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
        The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

        ### Target sentence:
        {data_point["ref"]}

        ### Meaning representation:
        {data_point["mr"]}
        """
        return tokenizer(full_prompt, truncation=True, max_length=self.model_max_length, padding='max_length')

    def load_and_prepare_data(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer,
            model_max_length=self.model_max_length,
            padding_side='left',
            add_eos_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenized_train_dataset, tokenized_val_dataset = self.prepare_datasets(
            lambda data_point: self.generate_and_tokenize_prompt(data_point, tokenizer),
        )
        return tokenized_train_dataset, tokenized_val_dataset, self.test_dataset

    @staticmethod
    def save_dataset(dataset, tag='train'):
        project_name, exec_id = get_run_identification()

        metadata = {
            'valohai.dataset-versions': [
                {
                    'uri': f'dataset://viggo/{project_name}_{tag}_{exec_id}',
                    'targeting_aliases': [f'dev_{tag}'],
                    'valohai.tags': ['dev', 'mistral'],
                },
            ],
        }
        out = valohai.outputs(f'encoded_{tag}')
        save_path = out.path('.')
        dataset.save_to_disk(save_path)

        for file in os.listdir(save_path):
            with open(out.path(f'{file}.metadata.json'), 'w') as outfile:
                json.dump(metadata, outfile)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Prepare data')

    # Add arguments based on your script's needs
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default='mistralai/Mistral-7B-v0.1', help='Huggingface tokenizer link')
    parser.add_argument('--model_max_length', type=int, default=512, help='Maximum length for the model')

    args = parser.parse_args()

    data_preprocessor = DataPreprocessor(args)

    tokenized_train_dataset, tokenized_val_dataset, test_dataset = data_preprocessor.load_and_prepare_data()

    data_preprocessor.save_dataset(tokenized_train_dataset, 'train')
    data_preprocessor.save_dataset(tokenized_val_dataset, 'val')
    data_preprocessor.save_dataset(test_dataset, 'test')


if __name__ == '__main__':
    main()
