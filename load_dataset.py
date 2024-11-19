import pandas as pd
import sklearn
from sklearn.decomposition import * 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
import numpy as np
import torch
from torch import nn

# 加载数据集
train_dataset = pd.read_csv(r'F:\pytorch_test\datase_unsw-nb2015\UNSW_NB15_training-set.csv')
test_dataset = pd.read_csv(r'F:\pytorch_test\datase_unsw-nb2015\UNSW_NB15_testing-set.csv')
print(train_dataset.shape) # (82332, 45)
print(test_dataset.shape) # (175341, 45)

# 合并数据集
combined_dataset = pd.concat([train_dataset, test_dataset]).drop(['id','label'],axis=1) #  合并训练集和测试集，并将标题为 id label 的列删除
print(combined_dataset.shape) # (257673, 43)
# print(combined_dataset.head(5))

# 计算"Normal"行的数量占总行数的比例
temp = train_dataset.where(train_dataset['attack_cat'] == "Normal").dropna() # 查找 标签=attack_cat的那一列，保留这一列中所有等于Normal的数据  保留所有Normal的行并删除其他行
print('Train_dataset:',round(len(temp['attack_cat'])/len(train_dataset['attack_cat']),5)) # 计算"Normal"行的数量占总行数的比例，并将结果四舍五入到小数点后5位。

temp = test_dataset.where(test_dataset['attack_cat'] == "Normal").dropna()
print("test_dataset:",round(len(temp['attack_cat'])/len(test_dataset['attack_cat']),5))

# 处理分类数据
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # 用于讲类别标签转换为数值
vector = combined_dataset['attack_cat'] # 提取attack_cat列
# vector_1 = combined_dataset['proto']
print("attack_cat:",list(set(list(vector))))
# print("proto:",list(set(list(vector_1))))
# 使用LabelEncoder将'attack_cat'列的文本标签转换为数值，并更新combined_data数据框中的相应列
combined_dataset['attack_cat'] = le.fit_transform(vector)
# 对'proto'、'service'和'state'这三列也执行相同的标签编码过程。
combined_dataset['proto'] = le.fit_transform(combined_dataset['proto'])
combined_dataset['service'] = le.fit_transform(combined_dataset['service'])
combined_dataset['state'] = le.fit_transform(combined_dataset['state'])

# 打印出描述攻击类型的信息，包括'attack_cat'列的众数（最常出现的值）
print("mode",vector.mode())
# print("mode_1",vector_1.mode())
# 计算数值为6的'attack_cat'标签所占的百分比，并打印出来。这里value=6指的是Normal
print(f"mode {np.sum(combined_dataset['attack_cat'].values==6)/vector.shape[0]:.2f}%")

# 使用了Python的collections模块和tabulate库来统计vector中元素的出现次数，并以表格的形式打印出来。
import collections
from tabulate import tabulate
# 创建一个计数器对象，计算vector中每个元素出现的次数
cnt = collections.Counter(vector)
# 使用tabulate函数将计数器对象中的元素及其出现次数以表格的形式打印出来。headers参数指定了表格的列标题
print(tabulate(cnt.most_common(), headers= ['Type', 'Occurences']))

# 数据复制备份
combined_dataset_COPY = combined_dataset.copy(deep=True)
combined_dataset = combined_dataset_COPY
assert combined_dataset_COPY.shape == combined_dataset.shape
# 计算标准差
# 整个表达式的意思是：找出combined_dataset数据框中标准差最小的7列，并返回这些列的列名。
# 这在特征选择时很有用，因为你可能想要排除那些变化不大（即标准差小）的特征。
# 计算combined_data中每列的标准差，并将标准差最小的7列的列名存储在lowSTD列表中。
# std() 计算标准差   to_frame() 转换为数据框   nsmallest() 返回数据框中最小的7个值所在的行。columns=0参数指定了要在哪一列中查找最小值，这里是第0列，也就是std()计算出的标准差列。
lowSTD = list(combined_dataset.std().to_frame().nsmallest(7, columns=0).index)
# 找出与’attack_cat’特征相关性最低的7个特征
# 使用数据框架（DataFrame）的corr()方法计算所有特征之间的相关性矩阵
# corr().abs()将相关性矩阵中的所有值转换为绝对值
# sort_values('attack_cat')根据'attack_cat'列的值对矩阵进行排序
# ['attack_cat'].nsmallest(7)找出与'attack_cat'相关性最小的7个值
# .index获取这些值的索引，即特征的名称
lowCORR = list(combined_dataset.corr().abs().sort_values('attack_cat')['attack_cat'].nsmallest(7).index)
print(lowSTD)
print(lowCORR)

# PCA降维
exclude = list(lowSTD + lowCORR)
# 移除  如果'attack_cat'在列表中，则将其移除
if 'attack_cat' in exclude:
    exclude.remove('attack_cat')
# 打印当前数据的形状
print('shape before:', combined_dataset.shape) # 43个特征
print('replace the following with their PCA(3) -', exclude) # 14个特征 7 + 7
# 使用PCA对选定的特征进行降维，降到3个维度
pca = PCA(3)
# 使用fit_transform方法对combined_dataset数据集中的exclude列表中的特征进行PCA降维。
dim_reduct = pca.fit_transform(combined_dataset[exclude])
# 打印PCA的方差比率之和，表示降维的信息量
print("explained_variance_ratio_ is", sum(pca.explained_variance_ratio_))
# 从原始数据中删除被PCA替换的特征
combined_dataset.drop(exclude, axis=1, inplace=True)
# 将降维后的数据转换为DataFrame
dim_reduction = pd.DataFrame(dim_reduct)
# 将降维后的数据与原始数据合并
combined_dataset = combined_dataset.join(dim_reduction)
# 打印合并后数据的形状
print('shape after:', combined_dataset.shape) # 43 - 14 + 3 = 32 个特征
# print("combined_data:",combined_dataset)

# 保存csv文件
# combined_dataset.to_csv('pca_dataset.csv', index=False)
print('combined_data.dur is scaled up by 10,000')
# 将 dur 列的值放大10000倍
combined_dataset['dur'] = 10000*combined_dataset['dur']
# combined_dataset.head()

# 准备机器学习模型的输入特征（data_x）和目标变量（data_y），并进行归一化处理
print('before:', combined_dataset.shape)
# 从数据集中删除‘attack_cat’列，删除标签
data_x = combined_dataset.drop(['attack_cat'], axis=1) # droped label
# 将标签单独保存
data_y = combined_dataset['attack_cat']
print(data_x.shape)
print(data_y.shape)
# 对data_x中的每个特征进行归一化处理，使其值在0到1之间
# 这是通过将每个值减去该特征的最小值，然后除以该特征的范围（最大值 - 最小值）来实现的
# x代表DataFrame中的一列。
data_x = data_x.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# data_x.to_csv('data_x.csv', index=False)
# data_y.to_csv('data_y.csv', index=False)

# 将数据集分割为训练集和测试集
# test_size=.50表示测试集占总数据集的50%
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=.50, random_state=42)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)
# 决策树DTC     随机森林RFC     极端随机森林ETC
DTC = DecisionTreeClassifier() 
# 导入随机森林分类器，设置50个树，随机状态为1
RFC = RandomForestClassifier(n_estimators=50, random_state=1)
# 导入极端随机树分类器，设置75个树，使用基尼系数，自动选择特征，不使用bootstrap
# ETC = ExtraTreesClassifier(n_estimators=75, criterion='gini', max_features='None', bootstrap=False)
# 创建一个投票分类器，它将上述三个分类器的预测结果进行硬投票

# eclf = VotingClassifier(estimators=[('lr', DTC), ('rf', RFC)], voting='hard') 
# for clf, label in zip([DTC, RFC, eclf], ['DecisionTreeClassifier', 'RandomForestClassifier',  'Ensemble']): 
#     _ = clf.fit(X_train,y_train)
#     pred = clf.score(X_test,y_test)
#     print("Acc: %0.7f [%s]" % (pred,label))


from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA,TruncatedSVD,PCA
from sklearn.svm import LinearSVC

# 使用RFE方法减少特征的数量，只保留对模型预测最关键的10个特征，然后基于这些特征创建新的训练和测试数据集。这样做可以提高模型的性能，减少过拟合的风险，并可能减少模型训练的时间。
estimator = DecisionTreeClassifier()
rfe = RFE(estimator,n_features_to_select=10)
rfe.fit(X_train,y_train)
# rfe = RFE(DecisionTreeClassifier(), 10).fit(X_train,y_train)
desiredIndices = np.where(rfe.support_==True)[0]

X_train, X_test = pd.DataFrame(X_train),  pd.DataFrame(X_test)

whitelist = X_train.columns.values[desiredIndices]
X_train_RFE,X_test_RFE = X_train[whitelist],X_test[whitelist]
print("X_train_RFE 包含的特征:",X_train_RFE.columns)



# print(X_train_RFE.columns)
DTC = DecisionTreeClassifier(random_state=1, criterion='entropy',splitter='random') 
RFC = RandomForestClassifier(n_estimators=50, random_state=1)
# ETC = ExtraTreesClassifier(n_estimators=75, criterion='gini', max_features='auto', bootstrap=False)

# X_train.shape
# X_train_RFE.shape
# y_train.shape

# print()
# X_test.shape
# X_test_RFE.shape
# y_test.shape

eclf = VotingClassifier(estimators=[('lr', DTC), ('rf', RFC)], voting='hard') 
for clf, label in zip([DTC, RFC, eclf], ['DecisionTreeClassifier', 'RandomForestClassifier', 'Ensemble']): 
    _ = clf.fit(X_train,y_train)
    pred = clf.score(X_test,y_test)
    print("Acc: %0.7f [%s]" % (pred,label))
# from sklearn import tree
# import graphviz
# import pydotplus  
# dot_data = tree.export_graphviz(DT, out_file=None, 
#                          feature_names=x_train.columns,  
#                          class_names=['0','1'],  
#                          filled=True, rounded=True,  
#                          special_characters=True)  
# graph = pydotplus.graph_from_dot_data(dot_data)  
# graph.write_pdf("DTtree.pdf") 


# 搭建神经网络
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

input_size = 10
hidden_size = 64
hidden_size_2 = 64
num_classes = 10

epochs = 100
batch_size = 32
learning_rate = 0.01

from cnn_model import *
model = CNN(input_size,hidden_size,num_classes)
model = model.cuda()
# 损失函数与优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
# 训练模型
total_step = len(X_train)

# 保存训练集与测试集
X_test_RFE.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv',index=False)
X_train_RFE.to_csv('X_train.csv',index=False)
y_train.to_csv('y_train.csv',index=False)



X_train_RFE_vals = X_train_RFE.values
y_train_vals = y_train.values
X_test_RFE_vals= X_test_RFE.values
y_test_vals = y_test.values
# print(X_train_RFE_vals)
# print(y_train_vals)

# 定义一个列表来存储准确率数据
accuracy_data = []

for epoch in range(epochs):
    for i in range(0,X_train_RFE_vals.shape[0],batch_size):
        # x = torch.as_tensor(X_train_RFE_vals[i:i+batch_size],dtype=torch.float).to(device)
        # y = torch.as_tensor(y_train_vals[i:i+batch_size],dtype=torch.long).to(device)

        x = torch.as_tensor(X_train_RFE_vals[i:i+batch_size],dtype=torch.float)
        y = torch.as_tensor(y_train_vals[i:i+batch_size],dtype=torch.long)
        x = x.cuda()      
        y= y.cuda()
        outputs = model(x)
        loss = loss_fn(outputs,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    
    with torch.no_grad():
        n_correct = 0 # 正确预测数
        n_samples = 0 # 样本总数
        for i in range(0,X_test_RFE_vals.shape[0],batch_size):
            x = torch.as_tensor(X_test_RFE_vals[i:i+batch_size],dtype=torch.float)
            y = torch.as_tensor(y_test_vals[i:i+batch_size],dtype=torch.long)
            x = x.cuda()      
            y= y.cuda()

            outputs = model(x)
            if len(outputs.data) > 0:
                # predicted = torch.max(outputs.data,dim=1)
                predicted = torch.max(outputs.data, dim=1)[1]
                n_samples += y.size(0)
                n_correct += (predicted == y).sum().item()
            else :
                print("??")
                print(x,outputs.data)
                
        acc = 100.0 * n_correct / (n_samples)
        
        print(f'Accuracy of the network——{epoch+1}:{acc}%')
        # 将准确率数据添加到列表中
        accuracy_data.append({'Epoch': epoch + 1, 'Accuracy': acc})
        if (epoch+1) % 10 == 0:
            torch.save(model,"model_{}.pth".format(epoch+1))
            print("模型已保存")

# 将准确率数据转换为DataFrame
accuracy_df = pd.DataFrame(accuracy_data)

# 将DataFrame保存到CSV文件
accuracy_df.to_csv('accuracy.csv', index=False)

print("准确率数据已保存到CSV文件中。")