from datetime import datetime

# 记录训练开始时间
start_time = datetime.now()
print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

# 在训练中执行操作

# 训练结束后记录结束时间并计算总时间
end_time = datetime.now()
elapsed_time = end_time - start_time
print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
print("Elapsed time: ", elapsed_time)
