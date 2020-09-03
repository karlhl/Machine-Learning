import pandas as pd
import numpy as np
"""
cora.content: 2708行，表示样本点，表示一篇论文：论文编号、1433的二进制位、论文分类组成
cora.cites共5429行， 每一行有两个论文编号，表示第一个编号的论文先写，第二个编号的论文引用第一个编号的论文。

"""



raw_data = pd.read_csv('../data/cora/cora.content',sep = '\t',header = None)
num = raw_data.shape[0] # 样本点数2708
print(num)

# 将论文的编号转[0,2707]顺序编码
a = list(raw_data.index)
b = list(raw_data[0])
c = zip(b,a)
map = dict(c)

print(map)