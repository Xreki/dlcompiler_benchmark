#python3
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
os.environ['XLA_FLAGS'] = '--xla_hlo_profile --xla_dump_to=./tmp/generated'
os.environ['TF_CPP_VMODULE'] = 'xla_compilation_cache=1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'


def block(n,c,h,w):
    _input = tf.compat.v1.placeholder(tf.float32, [n,c,h,w])

    conv0 = tf.keras.layers.Conv2D(filters=128, kernel_size=1, data_format="channels_first")
    _conv0 = conv0(_input)

    bn0 = tf.keras.layers.BatchNormalization(axis=1)
    _bn0 = bn0(_conv0, training=True)

    relu0 = tf.keras.layers.ReLU()
    _relu0 = relu0(_bn0)

    conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", data_format="channels_first")
    _conv1 = conv1(_relu0)

    bn1 = tf.keras.layers.BatchNormalization(axis=1)
    _bn1 = bn1(_conv1, training=True)

    relu1 = tf.keras.layers.ReLU()
    _relu1 = relu1(_bn1)

    conv2 = tf.keras.layers.Conv2D(filters=c, kernel_size=1, padding="valid", data_format="channels_first")
    _conv2 = conv2(_relu1)

    bn2 = tf.keras.layers.BatchNormalization(axis=1)
    _bn2 = bn2(_conv2, training=True)

    relu2 = tf.keras.layers.ReLU()
    _relu2 = relu2(_bn2 + _input)

    return _input,_relu2

_input=None
_output = None

_input,_output = block(64, 512, 56, 56)

config = tf.compat.v1.ConfigProto()
config.graph_options.optimizer_options.global_jit_level=2
config.gpu_options.allow_growth=True
config.log_device_placement=True
config.allow_soft_placement=False

print(config)
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session(config=config)
sess.run(init)

data = np.random.rand(64, 512, 56, 56).astype(np.float32)
with tf.device("GPU:0"):
    for i in range(10):
        _0 = sess.run([_output], feed_dict = {_input:data})