from datasets import load_dataset


if __name__ == '__main__':
    dataset_name = 'Dahoas/rm-static'
    data = load_dataset(dataset_name)
    prompts = data['train']['prompt']
    unique_prompts = set(data['train']['prompt'])

    print(f'prompts num: {len(prompts)} unique_prompts num: {len(unique_prompts)}')
    # prompts num: 76256 unique_prompts num: 76094
    print('end')
