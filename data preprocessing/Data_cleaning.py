import itertools
import re
import json
import jsonlines
import psutil
import ujson
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset
import os
from tqdm import tqdm

bos_token = '<s>'
eos_token = '</s>'

# upload tokenizer
tokenizer = AutoTokenizer.from_pretrained('./tokenizer/trained_tokenizer', use_fast=False)
# print(len(tokenizer))

def main():
    # checking JSONL data
    valid_lines, invalid_lines = check_jsonl(file_path)
    print(f"检查完成，文件中共有 {valid_lines} 行有效的 JSON 数据，{invalid_lines} 行无效的 JSON 数据。")

    # preprocess pretraining data
    pretain_data_process()

def preview_data(file_path, lines=10):
    '''
    read lines lines words of file
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} 路径错误，文件不存在！")
    with jsonlines.open(file_path) as reader:
        for index,object in enumerate(reader):
            print(f"第 {index + 1} 行数据:{object}")
            if index + 1 >= lines:
                break

file_path = r'./dataset/mobvoi_seq_monkey_general_open_corpus.jsonl'
# preview_data(file_path)

def get_line(file_path):
    with open(file_path, 'rb') as f:
        return sum(1 for _ in f)

def check_jsonl(file_path):
    total_lines = get_line(file_path)
    valid_lines = 0
    invalid_lines = 0

    with open(file_path, 'rb') as f:
        for index, line in tqdm(enumerate(f), total=total_lines, desc="Checking jsonl progress"):
            try:
                decoded_line = line.decode('utf-8')
                object = jsonlines.Reader([decoded_line]).read()
                valid_lines += 1
            except UnicodeDecodeError as e:
                print(f"Encoding error at line {index + 1}: {e}")
                invalid_lines += 1
            except jsonlines.InvalidLineError as e:
                print(f"Invalid JSON at line {index + 1}: {e}")
                invalid_lines += 1
    return valid_lines, invalid_lines

# 删除无效行函数；并且转存
def remove_invalid_line(file_path, output_path, invalid_lines):
    if invalid_lines ==0:
        '''
        将file_path下的文件存到output_path下
        '''
        with open(file_path, 'rb')as f, open(output_path, 'wb') as w:
            w.write(f.read())
    else:
        with open(file_path, 'rb') as f, open(output_path, 'wb') as w:
            for index, line in enumerate(f):
                if index + 1 != invalid_lines:
                    w.write(line)

# 处理函数
def process_seq_monkey(chunk_size=50000):
    """
    逐块读取 mobvoi_seq_monkey_general_open_corpus.jsonl 文件，
    对文本进行分词，并将分词结果保存为二进制文件，支持跳过无效行，并显示处理进度。
    """
    doc_ids = []
    chunk_index = 0

    # 计算总行数
    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # 读取训练数据
    with jsonlines.open(file_path) as reader:
        with tqdm(total=total_lines, desc="Processing lines") as pbar:
            while True:
                try:
                    # itertools.islice 按块读取文件，每次读取 chunk_size 行数据
                    chunk = list(itertools.islice(reader, chunk_size))
                except jsonlines.InvalidLineError as e:
                    print(f"Skipping invalid chunk at chunk {chunk_index}: {e}")
                    continue

                if not chunk:
                    break

                '''
                遍历每一块中的每一行数据，按照token逐行编码
                '''
                for index, object in enumerate(chunk):
                    try:
                        content = object.get('text', '')
                        # 跳过长度超过512的文本
                        if len(content) > 512:
                            continue
                        text_id = tokenizer(f'{bos_token}{content}{eos_token}').data['input_ids']

                        doc_ids += text_id

                    except UnicodeDecodeError as e:
                        print(f"Skipping invalid chunk at chunk {chunk_index * chunk_size + index + 1}: {e}")
                        continue

                #update chunk_idx
                chunk_index += 1
                #更新进度条
                pbar.update(len(chunk))

                if len(doc_ids) >= 1000000:
                    arr = np.array(doc_ids, dtype=np.uint16)
                    with open(f'.\dataset\clean_seq_monkey.bin', 'ab') as f:
                        f.write(arr.tobytes())
                    doc_ids = []
    # 如果存在最后未保存的内容，再次保存
    if doc_ids:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(f'.\dataset\clean_seq_monkey.bin', 'ab') as f:
            f.write(arr.tobytes())

def pretain_data_process():
    '''
    调用处理函数生成数据，并整合所有二进制文件为一个统一的预训练data文件
    '''
    process_seq_monkey(50000)
    data_path_list = [r'.\dataset\clean_seq_monkey.bin'] #当前仅存在一个目录一个文件

    data_array = []

    for data_path in data_path_list:
        with open(data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_array.append(data)

    print(type(data_array))

    final_array = np.concatenate(data_array) # 合并为一个大数组
    print(f"预训练数据大小：{final_array.shape}, 当前数据类型：{type(final_array)}")

    with open('.\dataset\pretrain_data.bin', 'wb') as f:
        f.write(final_array.tobytes())


if __name__ == '__main__':
    main()
