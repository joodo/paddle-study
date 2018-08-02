import paddle
import paddle.fluid as fluid


# Reader

BATCH_SIZE = 64

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE,
)

test_reader = paddle.batch(
    paddle.dataset.mnist.test(),
    batch_size=BATCH_SIZE,
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
conv_pool_1_layer = fluid.layers.batch_norm(conv_pool_1_layer)

conv_pool_2_layer = fluid.nets.simple_img_conv_pool(
    input=conv_pool_1_layer,
    filter_size=5,
    num_filters=50,
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

batch_size_layer = fluid.layers.create_tensor(dtype='int64')
acc_layer = fluid.layers.accuracy(input=prediction_layer,
                                  label=label_layer,
                                  total=batch_size_layer)


# test program
inference_program = fluid.default_main_program().clone(for_test=True)


# Optimizer
optimizer= fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost_layer)


# Exe
use_cuda = False
with open('../use_gpu') as f:
	if int(f.read()) != 0:
		use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# Feeder
feeder = fluid.DataFeeder(place=place, feed_list=[img_layer, label_layer])


# Train!!
while True:
    for batch_id, data in enumerate(train_reader()):
        acc, = exe.run(feed=feeder.feed(data), fetch_list=[acc_layer])
        if batch_id % 100 == 0:
			print('batch %d, acc %f' % (batch_id, acc))

    avg_acc = fluid.metrics.Accuracy()
    for data in test_reader():
        acc, weight = exe.run(
            inference_program,
            feed=feeder.feed(data),
            fetch_list=[acc_layer, batch_size_layer],
        )
        avg_acc.update(acc, weight)
    print('Final acc: %f' % (avg_acc.eval(),))
    if avg_acc.eval() > 0.99:
        break

fluid.io.save_inference_model('saved_model', ['img'], [prediction_layer], exe)
print('Model "saved_model" saved.')
