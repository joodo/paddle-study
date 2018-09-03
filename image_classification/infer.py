import paddle.fluid as fluid
from PIL import Image
import numpy as np


ITEM = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Load Data

def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)

    im = np.array(im).astype(np.float32)
    im = im[:, :, :3]
    # The storage order of the loaded image is W(width),
    # H(height), C(channel). PaddlePaddle requires
    # the CHW order, so transpose them
    im = im.transpose((2, 0, 1))
    im = im / 255.0
    # Add one dimension to mimic the list format.
    im = np.expand_dims(im, axis=0)

    return im


img = load_image('image.bmp')


# Load Model

use_cuda = False
with open('../use_gpu') as f:
	if int(f.read()) != 0:
		use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

inference_scope = fluid.Scope()
with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names,fetch_targets] = \
        fluid.io.load_inference_model('saved_model', exe)
    results = exe.run(
        inference_program,
        feed={ feed_target_names[0]: img },
        fetch_list=fetch_targets,
    )
    results = results[0][0]
    results = [(ITEM[i], results[i] * 100) for i in xrange(10)]
    def f(x, y):
        if x[1] > y[1]:
            return -1
        if x[1] == y[1]:
            return 0
        if x[1] < y[1]:
            return 1
    results.sort(cmp=f)
    for i in results:
        print('%s: %02f%%' % i)
