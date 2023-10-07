from scipy.optimize import minimize

# 定义目标函数（示例：最小化函数 f(x) = (x - 2)^2）
def objective_function(x):
    return (x - 2)**2

# 设置变量的初始猜测值
x0 = [0.0]  # 初始值为 0.0

# 定义边界约束，这里假设 x 取值范围为 [1.0, 3.0]
bounds = [(1.0, 3.0)]

# 使用 L-BFGS-B 方法进行优化
result = minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds)

# 输出优化结果
print("Optimal solution:", result.x)
print("Optimal value:", result.fun)
