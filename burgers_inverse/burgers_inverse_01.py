import scipy

# 导入数据
data = scipy.io.loadmat('./data/Burgers.mat')
x = data['x']
t = data['t']
usol = data['usol']
print(usol)
