#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paddle.fluid as fluid
import paddle.v2 as paddle


def train():
    # 类别数量（0～9 共 10 个）
    class_dim = 10
    # 图像通道和大小
    image_shape = [1, 28, 28]

    # 输入层
    layer_image = fluid.layers.data(name='image', shape=image_shape, dtype='int8')
    # 输出层（标签层）
    layer_label = fluid.layers.data(name='label', shape=[1], dtype='int8')


if __name__ == '__main__':
    train()
