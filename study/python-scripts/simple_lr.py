#!/usr/bin/env python3

import paddle.fluid as fluid
import numpy as np

# 一个简单的线性方程组的求解
# from https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/quick_start_cn.html

np.random.seed(0)
np_data = np.random.randint(5, size=(10, 4))

# 假设方程式为 y = 4a + 6b + 7c + 2d
np_res = []
for i in range(10):
    y = 4 * np_data[i][0] + \
        6 * np_data[i][1] + \
        7 * np_data[i][2] + \
        2 * np_data[i][3]
    np_res.append([y])

# 当作训练数据源和label结果
train_data = np.array(np_data).astype('float32')
y_true = np.array(np_res).astype('float32')


# 定义网络结构
x = fluid.layers.data(name="x", shape=[4], dtype='float32')
y = fluid.layers.data(name="y", shape=[1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)

# 定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)

#定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05)
sgd_optimizer.minimize(avg_cost)

#参数初始化
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

##开始训练，迭代500次
for i in range(500):
    outs = exe.run(
        feed = {'x':train_data,'y':y_true},
        fetch_list = [y_predict.name, avg_cost.name])
    if i % 50 == 0:
        print('iter={:.0f},cost={}'.format(i, outs[1][0]))

#存储训练结果
params_dirname = "train_parameters"
fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

print("TRAIN AND SAVE MODEL DONE.")

print("LOAD MODEL.")
infer_exe = fluid.Executor(cpu)
inference_scope = fluid.Scope()
# 加载训练好的模型
with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names, fetch_targets] = \
        fluid.io.load_inference_model(params_dirname, infer_exe)

# 输入数据
input_data = np.array([[[9], [5], [2], [10]]]).astype('float32')
results = infer_exe.run(
    inference_program,
    feed={"x": input_data},
    fetch_list=fetch_targets)

print("9a+5b+2c+10d={}".format(results[0][0]))
print("DONE.")