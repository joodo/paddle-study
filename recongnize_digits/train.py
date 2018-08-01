import paddle
import paddle.fluid as fluid


# Reader

BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
    batch_size=64,
)

test_reader = paddle.batch(
    paddle.dataset.mnist.test(),
    batch_size=64,
)


# LeNet-5:
# Conv-Pool-BatchNormalization
# Conv-Pool
# FullConnection

img_layer = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
label_layer = fluid.layers.data(name='label', shape=[1], dtype='int64')

conv_pool_1_layer = fluid.nets.simple_img_conv_pool(
    input=img_layer,
    filter_size=5,
    num_filters=20,
    pool_size=2,
    pool_stride=2,
    act='relu',
)
norm_1_layer = fluid.layers.batch_norm(conv_pool_1_layer)

conv_pool_2_layer = fluid.nets.simple_img_conv_pool(
    input=norm_1_layer,
    filter_size=5,
    num_filters=20,
    pool_size=2,
    pool_stride=2,
    act='relu',
)

prediction_layer = fluid.layers.fc(
    input=conv_pool_2_layer,
    size=10,
    act='softmax',
)

cost_layer = fluid.layers.cross_entropy(input=prediction_layer, label=label_layer)
avg_cost_layer = fluid.layers.mean(cost_layer)
acc_layer = fluid.layers.accuracy(input=prediction_layer, label=label_layer)


# test program
inference_program = fluid.default_main_program().clone(for_test=True)


# Optimizer
optimizer= fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost_layer)


# Exe
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_main_program())


# Feeder
feeder = fluid.DataFeeder(place=place, feed_list=[img_layer, label_layer])


# Train!!
while True:
    for data in train_reader():
        exe.run(feed=feeder.feed(data), fetch_list=[avg_cost_layer, acc_layer])

    avg_acc = fluid.metrics.Accuracy()
    for data in test_reader():
        acc, = exe.run(
            inference_program,
            feed=feeder.feed(data),
            feed_list=[acc_layer],
        )
        avg_acc.ipdate(acc, 1)
    print(avg_acc.eval())
    raw_input('pause')
