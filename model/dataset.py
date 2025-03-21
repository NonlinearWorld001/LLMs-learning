import json
import random
import re # re是正则表达式库

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false" # 设置环境变量，禁用tokenizer并行化

class PretrainDataset(Dataset):
    def __init__(self, data_pth_lst, maxlength=512, memmap=False):
        '''
        data_pth_lst: 数据路径列表
        maxlength: 最大序列长度
        memmap: 是否使用内存映射
        '''
        super().__init__()
        if memmap: # 使用内存映射的时候，文件应当已经被合并成为一个超大文件
            with open(data_pth_lst[0], 'r') as f: # 打开第一个文件
                nbytes = f.seek(0, 2) # 将文件指针移动到文件末尾，.seek()接受两个参数，第一个表示指针偏移量，第二个表示指针基准位置，获取文件大小（总字节数）
                flen = f.tell() // np.dtype('uint16').itemsize
                # f.tell()返回文件指针当前位置
                # np.dtype('uint16').itemsize返回uint16类型的大小
                # 将文件大小除以uint16类型的大小，得到文件中元素的个数
                self.data = np.memmap(data_pth_lst[0], dtype=np.dtype('uint16'), shape=(flen // maxlength, maxlength))
                # np.memmap()创建一个内存映射对象，用于读取和写入内存中的数据
        else: # 不使用内存映射的时候，文件应当是多个小文件的组合序列
            data_lst = []
            for data_pth in data_pth_lst:
                with open(data_pth, 'rb') as f: # 'rb'表示以二进制读取模式打开文件
                    data = np.fromfile(f, dtype=np.dtype('uint16')) # 从文件中读取数据，并转换为uint16类型
                    data_lst.append(data) # 将数据添加到列表中
                data = np.concatenate(data_lst) # 将列表中的数据拼接成一个数组
                data = data[ : maxlength * int(len(data)/maxlength)] # 将数据裁剪为maxlength的倍数，前maxlength*int(len(data)/maxlength)个元素
                self.data = data.reshape(-1, maxlength) 

        print("memmap:{} train data.shape:{}".format(memmap, self.data.shape)) # 打印数据形状
        print("downloading finished.....")
        
    def __len__(self):
        return self.data.shape[0] # 返回数据集的长度
    
    def __getitem__(self, idx):
        sample = np.array(self.data[idx]) # 获取数据集中的一个样本
        x = sample[:-1].astype(np.int64) # 获取样本的前maxlength-1个元素
        y = sample[1:].astype(np.int64) # 获取样本的后maxlength-1个元素
        return torch.from_numpy(x), torch.from_numpy(y) # 将x和y转换为torch.Tensor类型
        

#处理用于微调的SFT数据集
class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, maxlength=1024, prompt_maxlength=512, answer_maxlength=512):
        super().__init__()
        self.df = df # 数据集
        self.tokenizer = tokenizer # 分词器
        self.maxlength = maxlength # 最大序列长度
        self.prompt_maxlength = prompt_maxlength # 提示最大长度
        self.answer_maxlength = answer_maxlength # 答案最大长度
        #
        self.padding = 0 # 填充id
        self.bos_id = self.tokenizer('<s>assistant').data['input_ids']
        # 在对话模型中，a要明确区分用户（user）和助手（assistant）的角色。此处通过 <s>assistant 标记助手的回复开始，帮助模型理解生成内容的边界。
        # self.tokenizer('<s>assistant') 使用分词器将 '<s>assistant' 转换为 token 序列,生成一个字典，字典的键为 'input_ids'，值为 token 序列
        # .data['input_ids'] 获取字典的 'input_ids' 键对应的值，即 token 序列:提取编码后的token序列


    def __len__(self):
        return len(self.df)
    
    def find_sublist_idx(self, sublist, mainlist) -> int:
        last_idx = -1
        for i in range(len(mainlist) - len(sublist)+1):
            if mainlist[i : i+len(sublist)] == sublist:
                last_idx = i
        return last_idx
