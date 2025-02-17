import numpy as np
import random
import json
from datasets import load_dataset
from tokenizers import (decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer)
import os

# 读取训练语料
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data =json.loads(line)
            yield data['text']

data_path = r'D:\个人\AI\LLMs\deepseek\deepseek_zhx_v 1.0\dataset\tokenizer_train.jsonl'
texts = read_text_from_file(data_path)

# for i, text in enumerate(texts):
#     print(i,text)
#     if i == 5: break

# 初始化tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

special_tokens = ["<unk>", "<s>", "</s>"]

# 设置训练器，设置special token
trainer = trainers.BpeTrainer(vocab_size=6400, special_token=special_tokens, show_progress=True,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

tokenizer.train_from_iterator(texts, trainer=trainer)

print("successfully trained")

# 解码器
tokenizer.decoder = decoders.ByteLevel()

# save tokenizer
tokenizer_dir = r'D:\个人\AI\LLMs\deepseek\deepseek_zhx_v 1.0\trained_tokenizer'
os.makedirs(tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer.model.save(r"D:\个人\AI\LLMs\deepseek\deepseek_zhx_v 1.0\trained_tokenizer")

# 配置文件
config = {
    "add_bos_token": False,
    "add_eos_token": False,
    "add_prefix_space": True,
    "added_tokens_decoder": {
        "0": {
            "content": "<unk>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
        "1": {
            "content": "<s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            },
        "2": {
            "content": "</s>",
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
            }
    },
    "bos_token": "<s>",
    "clean_up_tokenization_spaces": False,
    "eos_token": "</s>",
    "legacy": True,
    "model_max_length": 1000000000000000019884624838656,
    "pad_token": None,
    "sp_model_kwargs": {},
    "spaces_between_special_tokens": False,
    "tokenizer_class": "PreTrainedTokenizerFast",
    "unk_token": "<unk>",
    "use_default_system_prompt": False,
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
}

with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), 'w', encoding='utf-8') as config_f:
    json.dump(config, config_f, ensure_ascii=False, indent=4)
print("successfully saved")