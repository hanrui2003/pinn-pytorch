import numpy as np

# 定义多个函数
def add(x):
    return x + 1

def square(x):
    return x ** 2

def cube(x):
    return x ** 3

# 将多个函数组成一个列表
funcs = [add, square, cube]

# 将多个函数向量化
vectorized_funcs = np.vectorize(lambda f, x: f(x))

# 生成一组输入数据
x = np.array([1, 2, 3])

# 将多个函数作用于同一个参数
result = vectorized_funcs(funcs, x)

print(result)
