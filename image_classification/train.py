import sys

import paddle
import paddle.fluid as fluid

from vgg import vgg_bn_drop as vgg
from resnet import resnet_cifar10 as resnet


# Reader

BATCH_SIZE = 128

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=50000),
    batch_size=BATCH_SIZE
)

test_reader = paddle.batch(
    paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE
)


# Layers

data_shape = [3, 32, 32]

images_layer = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')
prediction_layer = resnet(images_layer)
# prediction_layer = vgg(images_layer)
label_layer = fluid.layers.data(name='label', shape=[1], dtype='int64')

cost_layer = fluid.layers.cross_entropy(input=prediction_layer, label=label_layer)
avg_cost_layer = fluid.layers.mean(cost_layer)

batch_size_layer = fluid.layers.create_tensor(dtype='int64')
acc_layer = fluid.layers.accuracy(input=prediction_layer,
                                  label=label_layer,
                                  total=batch_size_layer)


# test program
inference_program = fluid.default_main_program().clone(for_test=True)


# Optimizer
optimizer= fluid.optimizer.Adam(
    learning_rate=0.001,
    regularization=fluid.regularizer.L2Decay(0.01),
)
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
feeder = fluid.DataFeeder(place=place, feed_list=[images_layer, label_layer])


# Train!!
pass_id = 0
while True:
    for batch_id, data in enumerate(train_reader()):
        acc, = exe.run(feed=feeder.feed(data), fetch_list=[acc_layer])
        if batch_id % 100 == 0:
			print('\nBatch %d, acc %f' % (batch_id, acc))
        else:
            sys.stdout.write('.')
            sys.stdout.flush()

    avg_acc = fluid.metrics.Accuracy()
    for data in test_reader():
        acc, weight = exe.run(
            inference_program,
            feed=feeder.feed(data),
            fetch_list=[acc_layer, batch_size_layer],
        )
        avg_acc.update(acc, weight)
    print('\nPass %d acc: %f' % (pass_id, avg_acc.eval()))
    pass_id += 1
    if avg_acc.eval() > 0.8:
        break

fluid.io.save_inference_model('saved_model', ['pixel'], [prediction_layer], exe)
print('Model "saved_model" saved.')
