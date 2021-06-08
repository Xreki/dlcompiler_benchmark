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
import argparse
import numpy as np
import tensorflow as tf

os.environ['XLA_FLAGS'] = '--xla_hlo_profile --xla_dump_to=./tmp/generated'
os.environ['TF_CPP_VMODULE'] = 'xla_compilation_cache=1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

parser = argparse.ArgumentParser(description='xla op test!')
parser.add_argument(
    "--op",
    type=str,
    default="normalization",
    help="[reduce, element_wise, normalization, batch normalization]")
parser.add_argument(
    "--mode", type=str, default="eager", help="[eager, static]")
pargs = parser.parse_args()


# (1) {elementwise/broadcast -> elementwise/broadcast/injective/reduce}
# elementwise/broadcast -> elementwise/broadcast
@tf.function(experimental_compile=True)
def add_sub(x, y, z):
    out = x + y - z
    return out


# elementwise/broadcast -> injective
@tf.function(experimental_compile=True)
def add_transpose(x, y, perm):
    out = tf.transpose(x + y, perm=perm)
    return out


# elementwise/broadcast -> reduce
@tf.function(experimental_compile=True)
def add_sum(x, y, axis=0):
    out = tf.math.reduce_sum(x + y, axis=axis)
    return out


# (2) {injective -> elementwise/broadcast/injective/reduce}
# injective -> elementwise/broadcast
@tf.function(experimental_compile=True)
def transpose_add(x, y, perm):
    out = tf.transpose(x, perm=perm) + y
    return out


# injective -> injective
@tf.function(experimental_compile=True)
def transpose_reshape(x, perm, shape):
    out = tf.reshape(tf.transpose(x, perm=perm), shape=shape)
    return out


# injective -> reduce
@tf.function(experimental_compile=True)
def transpose_sum(x, perm, axis):
    out = tf.math.reduce_sum(tf.transpose(x, perm=perm), axis=axis)
    return out


# (3) {complex-out-fusable -> elementwise/broadcast/injective/reduce}
# complex-out-fusable -> elementwise/broadcast
@tf.function(experimental_compile=True)
def conv_add(x, filters, bias):
    out = tf.nn.conv2d(
        x,
        filters,
        strides=[1, 1, 2, 2],
        padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
        data_format="NCHW") + bias
    return out


# complex-out-fusable -> injective
@tf.function(experimental_compile=True)
def conv_transpose(x, filters, perm=[0, 2, 3, 1]):
    out = tf.transpose(
        tf.nn.conv2d(
            x,
            filters,
            strides=[1, 1, 2, 2],
            padding='VALID',
            data_format="NCHW"),
        perm)
    return out


# complex-out-fusable -> reduce
@tf.function(experimental_compile=True)
def conv_sum(x, filters, axis):
    out = tf.math.reduce_sum(
        tf.nn.conv2d(
            x,
            filters,
            strides=[1, 1, 2, 2],
            padding='VALID',
            data_format="NCHW"),
        axis=axis)
    return out


# (4) {reduce -> elementwise/broadcast/injective/reduce}
# reduce -> elementwise/broadcast
@tf.function(experimental_compile=True)
def sum_add(x, y, axis=0):
    out = tf.math.reduce_sum(x, axis=axis) + y


# reduce -> injective
@tf.function(experimental_compile=True)
def sum_reshape(x, axis, shape):
    out = tf.reshape(tf.math.reduce_sum(x, axis=axis), shape=shape)


# reduce -> injective
@tf.function(experimental_compile=True)
def sum_max(x, axis):
    out = tf.math.reduce_max(tf.math.reduce_sum(x, axis=axis))


def run_static(feed_dict, fetch_list):
    tf.config.optimizer.set_jit(True)
    config = tf.compat.v1.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = 2
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    config.allow_soft_placement = False

    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session(config=config)
    sess.run(init)

    with tf.device("GPU:0"):
        for i in range(10):
            _ = sess.run(fetch_list, feed_dict=feed_dict)


if pargs.mode == "static":
    tf.compat.v1.disable_eager_execution()
    if pargs.op == "add_relu":
        x = tf.compat.v1.placeholder(tf.float32, [1000, 100])
        y = tf.compat.v1.placeholder(tf.float32, [1000, 100])
        x_np = np.random.random([1000, 100]).astype("float32")
        y_np = np.random.random([1000, 100]).astype("float32")
        out = add_relu(x, y)
        feed_dict = {x: x_np, y: y_np}
        fetch_list = [out]
        run_static(feed_dict, fetch_list)
    else:
        sys.exit("op is unkown!")

if pargs.op == "add_sub":
    x_np = np.random.random([1000, 100]).astype("float32")
    y_np = np.random.random([1000, 100]).astype("float32")
    z_np = np.random.random([1000, 1]).astype("float32")
    out = add_sub(x_np, y_np, z_np)
elif pargs.op == "add_transpose":
    x_np = np.random.random([1000, 100]).astype("float32")
    y_np = np.random.random([1000, 100]).astype("float32")
    out = add_transpose(x_np, y_np, perm=[1, 0])
elif pargs.op == "add_sum":
    x_np = np.random.random([1000, 100]).astype("float32")
    y_np = np.random.random([1000, 100]).astype("float32")
    out = add_sum(x_np, y_np, axis=[0])
elif pargs.op == "transpose_add":
    x_np = np.random.random([100, 1000]).astype("float32")
    y_np = np.random.random([1000, 100]).astype("float32")
    out = transpose_add(x_np, y_np, perm=[1, 0])
elif pargs.op == "transpose_reshape":
    x_np = np.random.random([100, 1000]).astype("float32")
    out = transpose_reshape(x_np, perm=[1, 0], shape=[10, 10000])
elif pargs.op == "transpose_sum":
    x_np = np.random.random([100, 1000]).astype("float32")
    out = transpose_sum(x_np, perm=[1, 0], axis=[1])
elif pargs.op == "conv_add":
    x_np = np.random.random([16, 64, 512, 512]).astype("float32")
    w_np = np.random.random([3, 3, 64, 256]).astype("float32")
    biad_np = np.random.random([256, 1, 1]).astype("float32")
    out = conv_add(x_np, w_np, biad_np)
elif pargs.op == "conv_transpose":
    x_np = np.random.random([16, 64, 512, 512]).astype("float32")
    w_np = np.random.random([3, 3, 64, 256]).astype("float32")
    out = conv_transpose(x_np, w_np, perm=[0, 2, 3, 1])
elif pargs.op == "conv_sum":
    x_np = np.random.random([16, 64, 512, 512]).astype("float32")
    w_np = np.random.random([3, 3, 64, 256]).astype("float32")
    out = conv_sum(x_np, w_np, axis=[0, 2, 3])
elif pargs.op == "sum_add":
    x_np = np.random.random([1000, 100]).astype("float32")
    y_np = np.random.random([100]).astype("float32")
    out = sum_add(x_np, y_np, axis=[0])
elif pargs.op == "sum_reshape":
    x_np = np.random.random([1000, 100]).astype("float32")
    out = sum_reshape(x_np, axis=[1], shape=[10, 100])
elif pargs.op == "sum_max":
    x_np = np.random.random([1000, 100]).astype("float32")
    out = sum_max(x_np, axis=[1])
else:
    sys.exit("op is unkown!")
