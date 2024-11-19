import pandas as pd

data_frame = pd.read_csv(r'F:\pytorch_test\data_x.csv')
data_frame['dur'] = 10000*data_frame['dur']
print(data_frame.columns)
len = len(data_frame.columns)
print(len)