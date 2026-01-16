import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('c_feats.pkl', 'rb') as file:
    loaded_c_feats = pickle.load(file)

A = loaded_c_feats[0]
A = A.squeeze(0)
C, H, W = A.size()
A = A.view(C, H*W)
# print(A.size())
plt.hist(A.cpu().numpy(), bins=30, edgecolor='black')
plt.xlabel('Value')  # x轴标签
plt.ylabel('Frequency')  # y轴标签
plt.title('Distribution Histogram')  # 标题
plt.grid(axis='y', alpha=0.75)  # 添加网格线
plt.show()  # 显示图形


