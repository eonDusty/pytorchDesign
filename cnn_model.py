import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

input_size = 10
hidden_size = 64
hidden_size_2 = 64
num_classes = 10

epochs = 500
batch_size = 32
learning_rate = 0.001

class CNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes) :
        super(CNN,self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size_2)
        self.l3 = nn.Linear(hidden_size_2,num_classes)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self,x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        output = self.relu(output)
        output = self.l3(output)
        return output

if __name__ == '__main__':
    # 测试网络模型的正确性
    model_self = CNN(input_size, hidden_size, num_classes).to(device)
    input = torch.ones((64,10))
    output = model_self(input)
    print(output.shape)



# import torch
# from torch import nn

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'  # 如果您想在CPU上运行，可以取消注释这一行

# input_size = 10  # 文本数据的特征数量
# hidden_size = 64  # RNN隐藏层的单元数量
# num_classes = 10  # 输出的类别数量

# class TextRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(TextRNN, self).__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN层
#         self.fc = nn.Linear(hidden_size, num_classes)  # 全连接输出层

#     def forward(self, x):
#         # x的形状为(batch_size, sequence_length, input_size)
#         _, hidden = self.rnn(x)  # 获取最后一个时间步的隐藏状态
#         # hidden的形状为(num_layers, batch_size, hidden_size)，我们取最后一个层的隐藏状态
#         out = hidden[-1, :, :]
#         out = self.fc(out)
#         return out

# # 创建模型实例
# model = TextRNN(input_size, hidden_size, num_classes).to(device)

# # 打印模型结构
# print(model)
