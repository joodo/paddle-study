#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paddle.fluid as fluid
import paddle.v2 as paddle

# from resnet import layer_resnet_creator


def train(learning_rate, num_passes, BATCH_SIZE=128):
    # 类别数量（0～9 共 10 个）
    class_dim = 10
    # 图像通道和大小
    image_shape = [1, 28, 28]

    # 输入层
    layer_image = fluid.layers.data(name='image', shape=image_shape, dtype='int8')
    # 标签层
    layer_label = fluid.layers.data(name='label', shape=[1], dtype='int8')
    # ResNet 层
    # layer_resnet = layer_resnet_creator(layer_image, 32)
    # Softmax 层
    layer_predict = fluid.layers.fc(input=layer_image, size=class_dim, act='softmax')

    # 获取损失函数层
    layer_cost = fluid.layers.cross_entropy(input=layer_predict, label=layer_label)
    # 定义平均损失函数层
    layer_avg_cost = fluid.layers.mean(x=layer_cost)

    # 计算 batch 准确率
    batch_size = fluid.layers.create_tensor(dtype='int8')
    batch_acc = fluid.layers.accuracy(input=layer_predict, label=layer_label, total=batch_size)

    # 优化方法
    optimizer = fluid.optimizer.Momentum(
        learning_rate = learning_rate,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(5 * 1e-5),
    )
    opts = optimizer.minimize(layer_avg_cost)

    # 调试器
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 初始化
    exe.run(fluid.default_startup_program())

    # 训练数据 Reader
    reader_train = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=8192),
        patch_size=BATCH_SIZE,
    )

    # 数据对应关系
    feeder = fluid.DataFeeder(place=place, feed_list=[layer_image, layer_label])
    accuracy = fluid.average.WeightedAverage()

    # 开始训练
    for pass_id in range(num_passes):
        accuracy.reset()

        # 读 batch
        for batch_id, data in enumerate(reader_train()):
            print(batch_id+': '+data)
            # 训练！
            loss, acc, weight = exe.run(fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[layer_avg_cost, batch_acc, batch_size])
            accuracy.add(value=acc, weight=weight)
            print('Pass {0}, batch {1}, loss {2}, acc {3}'.format(
                pass_id, batch_id, loss[0], acc[0]
            ))


if __name__ == '__main__':
    train(learning_rate = 0.005, num_passes = 300)
