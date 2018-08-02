import os

import paddle.fluid as fluid
import numpy as np
from PIL import Image


# Load Data

def  load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im /255.0 * 2.0 - 1.0
    return im

img = load_image('timg.jpeg')


# Load Model

use_cuda = False
with open('../use_gpu') as f:
	if int(f.read()) != 0:
		use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

inference_scope = fluid.core.Scope()
with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names,fetch_targets] = \
        fluid.io.load_inference_model('saved_model', exe)
    results = exe.run(
        inference_program,
        feed={ feed_target_names[0]: img },
        fetch_list = fetch_targets,
    )
    results = np.argsort(-results[0])
    print(results)
