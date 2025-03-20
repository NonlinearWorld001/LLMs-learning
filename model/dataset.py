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
        
                
                
