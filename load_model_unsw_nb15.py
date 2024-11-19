import pandas as pd
import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Flatten
from torch.nn import Linear,Sequential
import matplotlib.pyplot as plt
# 搭建神经网络
from cnn_model import *


# 加载测试数据集
X_test = pd.read_csv('F:\\pytorch_test\\datase_unsw-nb2015\\X_test.csv')
# 加载神经网络模型
model = torch.load("F:\pytorch_test\model_UNSW_NB1540.pth")
model = model.cuda()
print(model)

# 数据处理
X_test_vals = X_test.values
dic = ['DoS', 'Normal', 'Shellcode', 'Reconnaissance', 'Worms', 'Analysis', 'Generic', 'Fuzzers', 'Backdoor', 'Exploits']
# print(dic[0])
# 使用列表来收集预测结果
predictions = []

with torch.no_grad():
    for i in range(0,X_test_vals.shape[0],1):
        x = torch.as_tensor(X_test_vals[i:i+1],dtype=torch.float)
        x = x.cuda()
        outputs = model(x)
        pred = torch.max(outputs.data,dim=1)[1]
        predictions.append(dic[pred[0].item()])
        # print(pred[0].item())
        # print(dic[pred[0].item()])
        # 将打印的数据存到csv文件中，列名为attack_cat
        # with open(r'F:\pytorch_test\datase_unsw-nb2015\pred_demo1.csv','a') as f:
        #     f.write(dic[pred[0].item()]+'\n')
        #     f.close()
# 创建一个pandas DataFrame，并将预测结果赋值给列'attack_cat'
df_predictions = pd.DataFrame(predictions, columns=['attack_cat'])

# 保存到CSV文件
df_predictions.to_csv(r'F:\pytorch_test\datase_unsw-nb2015\pred.csv', index=False)

# df = pd.read_csv(r'F:\pytorch_test\datase_unsw-nb2015\pred_demo1.csv')
# # 假设CSV文件中包含一个名为'attack_cat'的列，该列包含了您想要统计的单词
# word_counts = df['attack_cat'].value_counts()

# # 绘制柱状图
# word_counts.plot(kind='bar')

# # 设置图表标题和轴标签
# plt.title('Word Counts in CSV File')
# plt.xlabel('Words')
# plt.ylabel('Counts')

# # 显示图形
# plt.show()
