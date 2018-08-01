import paddle
import paddle.fluid as fluid
import numpy


BATCH_SIZE = 20

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500),
    batch_size=BATCH_SIZE,
)

test_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.uci_housing.test(), buf_size=500),
    batch_size=BATCH_SIZE,
)


def train_program():
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    loss = fluid.layers.square_error_cost(input=y_predict, label=y)
    #avg_loss = fluid.layers.mean(loss)

    return loss
    #return avg_loss


def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=0.001)


use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()


# trainer = fluid.Trainer(
#     train_func=train_program,
#     place=place,
#     optimizer_func=optimizer_program,
# )
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program)


step = 0
def event_handler(event):
    global step
    if isinstance(event, fluid.EndStepEvent):
        if event.step % 10 == 0:
            test_metrics = trainer.test(reader=test_reader, feed_order=feed_order)
            print(test_metrics)
            #raw_input('pause')

            if test_metrics[0] < 10.0:
                print('loss less than 10.0. stop')
                trainer.stop()

        step += 1


feed_order=['x', 'y']


trainer.train(
    reader=train_reader,
    event_handler=event_handler,
    num_epochs=100,
    feed_order=feed_order,
)
