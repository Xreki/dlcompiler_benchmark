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

parser = argparse.ArgumentParser(description='xla op test!')
parser.add_argument("--op", type=str, default="normalization", help="[reduce, element_wise, normalization, batch normalization]")
parser.add_argument("--axis", type=int, default=0, help="reduce axis!")
pargs = parser.parse_args()

def batch_n():
    _input = tf.compat.v1.placeholder(tf.float32, [None, 256, 32, 32])
    _moving_mean = tf.compat.v1.placeholder(tf.float32, [1,256])
    _moving_var = tf.compat.v1.placeholder(tf.float32, [1,256])

    _input_t = tf.transpose(_input, [0, 2, 3, 1])
    _input_t_flat = tf.reshape(_input_t, [-1, 256])
    _mean = tf.compat.v1.reduce_mean(_input_t_flat, axis=0, keepdims=True)
    _diff = _input_t_flat - _mean
    _diff2 = _diff*_diff
    _var = tf.compat.v1.reduce_mean(_diff2, axis=0, keepdims=True)
    _std_var = tf.math.sqrt(_var)
    _normal = _diff/_std_var
    _normal_r = tf.reshape(_normal, [-1, 32, 32, 256])
    _output = tf.transpose(_normal_r, [0, 3, 1, 2])

    _moving_mean_update = _moving_mean * 0.9 + _mean * 0.1
    _moving_var_update = _moving_var * 0.9 + _var * 0.1
    #_output = tf.compat.v1.layers.batch_normalization(_input, axis = 1, training = True)

    return (_input, _moving_mean, _moving_var), (_output, _moving_mean_update, _moving_var_update)

def normalization(axis):
    _input = tf.compat.v1.placeholder(tf.float32, [None, 1024, 256])
    _mean = tf.compat.v1.reduce_mean(_input, axis=axis, keepdims=True)
    _diff = _input - _mean
    _diff2 = _diff*_diff
    _mean2 = tf.compat.v1.reduce_mean(_diff2, axis=axis, keepdims=True)
    _var = tf.math.sqrt(_mean2)
    _output = _diff/_var
    return _input, _output

def reduce(axis):
    _input =  tf.compat.v1.placeholder(tf.float32, [128, 1024, 256])
    _output = tf.compat.v1.reduce_mean(_input, axis=axis, keepdims=True)
    return _input, _output

def element_wise():
    c, h, w = 256, 512, 1024
    A =  tf.compat.v1.placeholder(tf.float32, [c, h, w])
    B =  tf.compat.v1.placeholder(tf.float32, [1, h, w])
    C =  tf.compat.v1.placeholder(tf.float32, [c, 1, w])
    D =  tf.compat.v1.placeholder(tf.float32, [c, h, 1])

    _output = ((A + B) - C) * D
    return (A,B,C,D),_output

def element_wise_no_broadcast():
    c, h, w = 256, 512, 1024
    A =  tf.compat.v1.placeholder(tf.float32, [c, h, w])
    B =  tf.compat.v1.placeholder(tf.float32, [c, h, w])
    C =  tf.compat.v1.placeholder(tf.float32, [c, h, w])
    D =  tf.compat.v1.placeholder(tf.float32, [c, h, w])

    _output = ((A + B) - C) * D
    return (A,B,C,D),_output

_input=None
_output = None
if pargs.op == "normalization":
    _input, _output = normalization(pargs.axis)
elif pargs.op == "reduce":
    _input, _output = reduce(pargs.axis)
elif pargs.op == "element_wise":
    _input, _output = element_wise()
elif pargs.op == "element_wise_no_broadcast":
    _input, _output = element_wise_no_broadcast()
elif pargs.op == "batch_normalization":
    _input, _output = batch_n();
else:
    sys.exit("op is unkown!")

#print(tf.compat.v1.get_default_graph().ToGraphDefDebug())

#optimizer = tf.OptimizerOptions(global_jit_level=2, opt_level=0)
#gpu = tf.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto()
config.graph_options.optimizer_options.global_jit_level=2
config.gpu_options.allow_growth=True
config.log_device_placement=True
config.allow_soft_placement=False

print(config)
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session(config=config)
sess.run(init)

#print(sess.graph_def())
#tf.ops.get_default_graph().ToGraphDefDebug().toString()
if pargs.op == "normalization":
    data = np.random.rand(128, 1024, 256).astype(np.float32)
    with tf.device("GPU:0"):
        for i in range(10):
            _ = sess.run([_output], feed_dict = {_input : data})
elif pargs.op == "reduce":
    data = np.random.rand(128, 1024, 256).astype(np.float32)
    with tf.device("GPU:0"):
        for i in range(10):
            _ = sess.run([_output], feed_dict = {_input : data})
elif pargs.op == "element_wise":
    c,h,w = 256, 512, 1024
    dataA = np.random.rand(c, h, w).astype(np.float32)
    dataB = np.random.rand(1, h, w).astype(np.float32)
    dataC = np.random.rand(c, 1, w).astype(np.float32)
    dataD = np.random.rand(c, h, 1).astype(np.float32)

    with tf.device("GPU:0"):
        for i in range(10):
            _ = sess.run([_output], feed_dict = {_input[0] : dataA, _input[1] : dataB, _input[2] : dataC, _input[3] : dataD})
elif pargs.op == "element_wise_no_broadcast":
    c,h,w = 256, 512, 1024
    dataA = np.random.rand(c, h, w).astype(np.float32)
    dataB = np.random.rand(c, h, w).astype(np.float32)
    dataC = np.random.rand(c, h, w).astype(np.float32)
    dataD = np.random.rand(c, h, w).astype(np.float32)

    with tf.device("GPU:0"):
        for i in range(10):
            _ = sess.run([_output], feed_dict = {_input[0] : dataA, _input[1] : dataB, _input[2] : dataC, _input[3] : dataD})
elif pargs.op == "batch_normalization":
    data = np.random.rand(128, 256, 32, 32).astype(np.float32)
    mean = np.random.rand(1, 256).astype(np.float32)
    var = np.random.rand(1, 256).astype(np.float32)
    with tf.device("GPU:0"):
        for i in range(10):
            _0,_1,_2 = sess.run([_output[0], _output[1], _output[2]], feed_dict = {_input[0] : data, _input[1] : mean, _input[2] : var})

