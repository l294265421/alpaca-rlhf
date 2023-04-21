from transformers import AutoTokenizer


def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


if __name__ == '__main__':
    # /root/autodl-tmp/models/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348
    base_model: str = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    vocab = tokenizer.get_vocab()
    word_num = 100
    first_words = list(sorted(vocab.items(), key=lambda x: x[1]))[:word_num]
    for word in first_words:
        print(word)
