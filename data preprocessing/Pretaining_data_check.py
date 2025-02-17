import pandas as pd
import numpy as np

file_path = r'D:\个人\AI\LLMs\deepseek\deepseek_zhx_v 1.0\dataset\clean_seq_monkey.bin'
with open(file_path, 'rb') as f:
    data = np.fromfile(f, dtype=np.uint16)

data_str = data.astype(str)
df = pd.DataFrame(data_str, columns=['input_ids'])
csv_file_path = r'D:\个人\AI\LLMs\deepseek\deepseek_zhx_v 1.0\dataset\clean_seq_monkey.csv'
df.to_csv(csv_file_path, index=False)

pretrain_data = pd.read_csv(r'D:\个人\AI\LLMs\deepseek\deepseek_zhx_v 1.0\dataset\clean_seq_monkey.csv')

print(pretrain_data.head(10))