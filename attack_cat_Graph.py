from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(r'F:\pytorch_test\datase_unsw-nb2015\pred_demo_100.csv')
# 假设CSV文件中包含一个名为'attack_cat'的列，该列包含了您想要统计的单词
word_counts = df['attack_cat'].value_counts()

# # 绘制柱状图
# word_counts.plot(kind='bar')

# 创建图形并设置大小
plt.figure(figsize=(6, 4))  # 设置窗口宽度为10英寸，高度为6英寸

# 获取单词的索引位置，并将其转换为浮点数数组
indices = np.arange(len(word_counts))  # 创建一个从0到len(word_counts)的数组
colors = plt.cm.viridis(np.linspace(0, 1, len(word_counts)))

# 绘制柱状图，并指定颜色
word_counts.plot(kind='bar', color=colors)

plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['font.size'] = 10  # 字体大小为10，相当于五号字体
# plt.rcParams['font.weight'] = 'bold'  # 字体加粗
plt.rcParams['text.color'] = 'red'  # 字体颜色为红色

# 设置图表标题和轴标签
plt.title('攻击数据统计')
plt.xlabel('攻击类别')
plt.ylabel('数量')

# 自动调整图形大小
plt.tight_layout()

# 保存图片
plt.savefig('my_figure_1.png')

# 显示图形
plt.show()

# 创建图形并设置大小
plt.figure(figsize=(5, 5))  # 设置窗口宽度为8英寸，高度为8英寸

# 绘制饼状图
plt.pie(word_counts, labels=word_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.viridis(np.linspace(0, 1, len(word_counts))))

# 设置图表标题

plt.title('ATTACK_CAT ANALYSIS', fontsize=16, fontweight='bold', color='blue')

# 显示图形
plt.tight_layout()
# 保存图片
plt.savefig('my_figure_2.png')
plt.show()