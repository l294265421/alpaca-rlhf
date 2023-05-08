from datasets import load_dataset
from datasets.load import LocalDatasetModuleFactoryWithoutScript
from datasets import load_from_disk
from datasets import arrow_dataset
from datasets import arrow_reader
from datasets import DatasetInfo
import datasets


if __name__ == '__main__':
    dataset_name = 'Dahoas/rm-static'
    # data = load_dataset(dataset_name)
    data = load_dataset("parquet", data_files={'train': '/root/.cache/huggingface/datasets/downloads/8d6740386b550dd7c47f6d541f208491d2c6835a3dca2acec1d707873eb34e52.parquet', 'test': '/root/.cache/huggingface/datasets/downloads/f33aa6e0b02bc09b14306ff429145995450bf731e9ae79aac5f34b70241b5263.parquet'})
    prompts = data['train']['prompt']
    unique_prompts = set(data['train']['prompt'])

    print(f'prompts num: {len(prompts)} unique_prompts num: {len(unique_prompts)}')
    # prompts num: 76256 unique_prompts num: 76094
    print('end')
