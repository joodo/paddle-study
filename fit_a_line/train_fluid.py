import paddle
import paddle.fluid as fluid


BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE,
)

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.test(), buf_size=500),
    batch_size=BATCH_SIZE,
)


y = fluid.layers.data(name='y', shape=[1], dtype='float32')

x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)

loss_layer = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_loss_layer = fluid.layers.mean(loss_layer)


inference_program = fluid.default_main_program().clone(for_test=True)


optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_loss_layer)


use_cuda = False
with open('../use_gpu') as f:
	if int(f.read()) != 0:
		use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()


exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


feeder = fluid.DataFeeder(place=place, feed_list=[x, y])

test_loss = fluid.metrics.Accuracy()
while True:
    for batch_id, data in enumerate(train_reader()):
        avg_loss, = exe.run(
            feed=feeder.feed(data),
            fetch_list=[avg_loss_layer],
        )
    test_loss.reset()
    for data in test_reader():
        avg_loss, = exe.run(inference_program,
                       feed=feeder.feed(data),
                       fetch_list=[avg_loss_layer])
        test_loss.update(avg_loss, 1)
    print(test_loss.eval())
    if test_loss.eval() < 15.0:
        break

fluid.io.save_inference_model('fit_a_line_model', ['x'], [y_predict], exe)
print('Model "fit_a_line_model" saved.')
