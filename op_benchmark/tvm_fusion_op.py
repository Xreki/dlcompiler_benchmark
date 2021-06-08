# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tvm
import argparse
import numpy as np
from tvm import te
from tvm import topi
from tvm import auto_scheduler

import sys
sys.setrecursionlimit(100000)

parser = argparse.ArgumentParser(description='tvm op test!')
parser.add_argument(
    "--op",
    type=str,
    default="normalization",
    help="[reduce, element_wise, normalization]")
pargs = parser.parse_args()


# (1) {elementwise/broadcast -> elementwise/broadcast/injective/reduce}
# elementwise/broadcast -> elementwise/broadcast
@auto_scheduler.register_workload
def add_sub(h1, w1, h2, w2, h3, w3):
    x = te.placeholder((h1, w1), name='x')
    y = te.placeholder((h2, w2), name='y')
    z = te.placeholder((h3, w3), name='z')
    out = topi.subtract(topi.add(x, y), z)
    return [x, y, z, out]


# elementwise/broadcast -> injective
@auto_scheduler.register_workload
def add_transpose(h1, w1, h2, w2, perm=[1, 0]):
    x = te.placeholder((h1, w1), name='x')
    y = te.placeholder((h2, w2), name='y')
    out = topi.transpose(topi.add(x, y), perm)
    return [x, y, out]


# elementwise/broadcast -> reduce
@auto_scheduler.register_workload
def add_sum(h1, w1, h2, w2, axis=0):
    x = te.placeholder((h1, w1), name='x')
    y = te.placeholder((h2, w2), name='y')
    out = topi.sum(x + y, axis=axis)
    return [x, y, out]


# (2) {injective -> elementwise/broadcast/injective/reduce}
# injective -> elementwise/broadcast
@auto_scheduler.register_workload
def transpose_add(h1, w1, h2, w2, perm):
    x = te.placeholder((h1, w1), name='x')
    y = te.placeholder((h2, w2), name='y')
    out = topi.transpose(x, perm) + y
    return [x, y, out]


# injective -> injective
@auto_scheduler.register_workload
def transpose_reshape(h, w, perm, shape):
    x = te.placeholder((h, w), name='x')
    out = topi.reshape(topi.transpose(x, perm), shape)
    return [x, out]


# injective -> reduce
@auto_scheduler.register_workload
def transpose_sum(h, w, perm, axis):
    x = te.placeholder((h, w), name='x')
    out = topi.sum(topi.transpose(x, perm), axis)
    return [x, out]


# (3) {complex-out-fusable -> elementwise/broadcast/injective/reduce}
# complex-out-fusable -> elementwise/broadcast
@auto_scheduler.register_workload
def conv_add(n, c, h, w, out_c):
    x = te.placeholder((n, c, h, w), name='x')
    filters = te.placeholder((out_c, c, 3, 3), name='filters')
    bias = te.placeholder((out_c, 1, 1), name='bias')
    out = topi.nn.conv2d(
        x, filters, strides=[2, 2], padding=[0, 0], dilation=1,
        layout='NCHW') + bias
    return [x, filters, bias, out]


# complex-out-fusable -> injective
@auto_scheduler.register_workload
def conv_transpose(n, c, h, w, out_c, perm=[0, 2, 3, 1]):
    x = te.placeholder((n, c, h, w), name='x')
    filters = te.placeholder((out_c, c, 3, 3), name='filters')
    out = topi.transpose(
        topi.nn.conv2d(
            x,
            filters,
            strides=[2, 2],
            padding=[0, 0],
            dilation=1,
            layout='NCHW'),
        perm)
    return [x, filters, out]


# complex-out-fusable -> reduce
@auto_scheduler.register_workload
def conv_sum(n, c, h, w, out_c, axis):
    x = te.placeholder((n, c, h, w), name='x')
    filters = te.placeholder((out_c, c, 3, 3), name='filters')
    out = topi.sum(topi.nn.conv2d(
        x, filters, strides=[2, 2], padding=[0, 0], dilation=1, layout='NCHW'),
                   axis=axis)
    return [x, filters, out]


# (4) {reduce -> elementwise/broadcast/injective/reduce}
# reduce -> elementwise/broadcast
@auto_scheduler.register_workload
def sum_add(h1, w1, w2, axis=0):
    x = te.placeholder((h1, w1), name='x')
    y = te.placeholder((w2, ), name='y')
    out = topi.sum(x, axis=axis) + y
    return [x, y, out]


# reduce -> injective
@auto_scheduler.register_workload
def sum_reshape(h, w, axis, shape):
    x = te.placeholder((h, w), name='x')
    out = topi.reshape(topi.sum(x, axis=axis), shape)
    return [x, out]


# reduce -> injective
@auto_scheduler.register_workload
def sum_max(h, w, axis):
    x = te.placeholder((h, w), name='x')
    out = topi.max(topi.sum(x, axis=axis))
    return [x, out]


def get_task(func, args):
    task = auto_scheduler.SearchTask(func=func, args=args, target=target)
    return task


def tune_and_run(input_args, measure_ctx):
    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

    # Kill the measurement process
    del measure_ctx

    mod = tvm.build(sch, args, target)

    for i in range(10):
        mod(*input_args)

    print("CUDA source code:")
    print(task.print_best(log_file, print_mode="cuda"))


def get_input(shape):
    np_data = np.random.uniform(size=shape).astype(np.float32)
    tvm_input = tvm.nd.array(np_data, device=dev)
    return tvm_input


target = tvm.target.Target("cuda")
log_file = "%s.json" % pargs.op
dev = tvm.gpu()

measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2, )

if pargs.op == "add_sub":
    task = get_task(add_sub, [1000, 100, 1000, 100, 1000, 1])
    tvm_out = tvm.nd.empty(shape=[1000, 100], device=dev)
    args = [
        get_input([1000, 100]), get_input([1000, 100]), get_input([1000, 1]),
        tvm_out
    ]
    tune_and_run(args, measure_ctx)
elif pargs.op == "add_transpose":
    task = get_task(add_transpose, [1000, 100, 1000, 100])
    tvm_out = tvm.nd.empty(shape=[100, 1000], device=dev)
    args = [get_input([1000, 100]), get_input([1000, 100]), tvm_out]
    tune_and_run(args, measure_ctx)
elif pargs.op == "add_sum":
    task = get_task(add_sum, [1000, 100, 1000, 100, [0]])
    tvm_out = tvm.nd.empty(shape=[100], device=dev)
    args = [get_input([1000, 100]), get_input([1000, 100]), tvm_out]
    tune_and_run(args, measure_ctx)
elif pargs.op == "transpose_add":
    task = get_task(transpose_add, [100, 1000, 1000, 100, [1, 0]])
    tvm_out = tvm.nd.empty(shape=[1000, 100], device=dev)
    args = [get_input([100, 1000]), get_input([1000, 100]), tvm_out]
    tune_and_run(args, measure_ctx)
elif pargs.op == "transpose_reshape":
    task = get_task(transpose_reshape, [100, 1000, [1, 0], [10, 10000]])
    tvm_out = tvm.nd.empty(shape=[10, 10000], device=dev)
    args = [get_input([100, 1000]), tvm_out]
    tune_and_run(args, measure_ctx)
elif pargs.op == "transpose_sum":
    task = get_task(transpose_sum, [100, 1000, [1, 0], [1]])
    tvm_out = tvm.nd.empty(shape=[1000], device=dev)
    args = [get_input([100, 1000]), tvm_out]
    tune_and_run(args, measure_ctx)
elif pargs.op == "conv_add":
    task = get_task(conv_add, [16, 64, 512, 512, 256])
    tvm_out = tvm.nd.empty(shape=[16, 256, 255, 255], device=dev)
    args = [
        get_input([16, 64, 512, 512]), get_input([256, 64, 3, 3]),
        get_input([256, 1, 1]), tvm_out
    ]
    tune_and_run(args, measure_ctx)
elif pargs.op == "conv_transpose":
    task = get_task(conv_transpose, [16, 64, 512, 512, 256])
    tvm_out = tvm.nd.empty(shape=[16, 255, 255, 256], device=dev)
    args = [
        get_input([16, 64, 512, 512]), get_input([256, 64, 3, 3]),
        get_input([256, 1, 1]), tvm_out
    ]
    tune_and_run(args, measure_ctx)
elif pargs.op == "conv_sum":
    task = get_task(conv_sum, [16, 64, 512, 512, 256, [0, 2, 3]])
    tvm_out = tvm.nd.empty(shape=[256], device=dev)
    args = [
        get_input([16, 64, 512, 512]), get_input([256, 64, 3, 3]),
        get_input([256, 1, 1]), tvm_out
    ]
    tune_and_run(args, measure_ctx)
elif pargs.op == "sum_add":
    task = get_task(sum_add, [1000, 100, 1, [0]])
    tvm_out = tvm.nd.empty(shape=[100, ], device=dev)
    args = [get_input([1000, 100]), get_input([1]), tvm_out]
    tune_and_run(args, measure_ctx)
elif pargs.op == "sum_reshape":
    task = get_task(sum_reshape, [1000, 100, [1], [10, 100]])
    tvm_out = tvm.nd.empty(shape=[10, 100], device=dev)
    args = [get_input([1000, 100]), tvm_out]
    tune_and_run(args, measure_ctx)
elif pargs.op == "sum_max":
    task = get_task(sum_max, [1000, 100, [1]])
    tvm_out = tvm.nd.empty(shape=[], device=dev)
    args = [get_input([1000, 100]), tvm_out]
    tune_and_run(args, measure_ctx)
else:
    sys.exit("op is unkown!")
