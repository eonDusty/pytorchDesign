import torch
from torch import nn
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Flatten
from torch.nn import Linear,Sequential

# 搭建神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(), # 展平 -> 一维
            Linear(1024,64),
            Linear(64,10) # 
        )
    
    def forward(self,x):
        x = self.model1(x)
        return x
    
if __name__ == '__main__':
    # 测试网络模型的正确性
    modle_self = CNN()
    input = torch.ones((64,3,32,32))
    output = modle_self(input)
    print(output.shape)
    