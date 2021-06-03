#python3
import os
import sys
import tvm
import argparse
import numpy as np
from tvm import te
from tvm import topi
from tvm import auto_scheduler

parser = argparse.ArgumentParser(description='tvm op test!')
parser.add_argument("--op", type=str, default="normalization", help="[reduce, element_wise, normalization]")
parser.add_argument("--axis", type=int, default=0, help="reduce axis!")
pargs = parser.parse_args()

def tvm_batch_normalization(_input, n, c, h, w, name):
    _input_t = topi.transpose(_input, (0, 2, 3, 1))
    _input_t_flat = topi.reshape(_input_t, [-1, c])
    _input_sum = topi.sum(_input_t_flat, axis=0, keepdims=True)
    _input_mean = _input_sum / (n*h*w)
    _input_diff = topi.subtract(_input_t_flat, _input_mean)
    _input_diff2 = topi.multiply(_input_diff, _input_diff)
    _input_diff2_sum = topi.sum(_input_diff2, axis=0, keepdims=True)
    _input_var = _input_diff2_sum / (n*h*w)
    _input_std_var = topi.sqrt(_input_var)
    _input_normal = topi.divide(_input_diff, _input_std_var)
    _input_r = topi.reshape(_input_normal, (n, h, w, c))
    _output = topi.transpose(_input_r, (0, 3, 1, 2))

    _moving_mean = te.placeholder((1, c), name = name + '_mean')
    _moving_var = te.placeholder((1, c), name = name + '_var')
    _moving_mean_update = _moving_mean * 0.9 + _input_mean * 0.1
    _moving_var_update = _moving_var * 0.9 + _input_var * 0.1

    return [_moving_mean, _moving_var],[_moving_mean_update, _moving_var_update], _output #[_moving_mean, _moving_var, _output, _moving_mean_update, _moving_var_update]

@auto_scheduler.register_workload
def tvm_resnet_block(n, c, h, w):
    _input = te.placeholder((n, c, h, w), name = "_input")

    _filter1 =  te.placeholder((128, c, 1, 1), name = "filter1")#128 c 1 1
    _conv1 = topi.nn.conv2d_nchw(_input, _filter1, stride = 1, padding = 0, dilation = 1)
    _bn1_inputs, _bn1_outputs, _bn1 = tvm_batch_normalization(_conv1, n, 128, h, w, "_bn1")
    _relu1 = topi.nn.relu(_bn1)

    _filter2 =  te.placeholder((128, 128, 3, 3), name = "filter2")#128 128 3 3
    _conv2 = topi.nn.conv2d_nchw(_relu1, _filter2, stride = 1, padding = 1, dilation = 1)
    _bn2_inputs, _bn2_outputs, _bn2 = tvm_batch_normalization(_conv2, n, 128, h, w, "_bn2")
    _relu2 = topi.nn.relu(_bn2)

    _filter3 =  te.placeholder((c, 128, 1, 1), name = "filter3")#128 c 1 1
    _conv3 = topi.nn.conv2d_nchw(_relu2, _filter3, stride = 1, padding = 0, dilation = 1)
    _bn3_inputs, _bn3_outputs, _bn3 = tvm_batch_normalization(_conv3, n, c, h, w, "_bn3")
    _relu3 = topi.nn.relu(_bn3 + _input)

    return [_input,
            _filter1,_filter2,_filter3,
            _bn1_inputs[0],_bn1_inputs[1],
            _bn2_inputs[0],_bn2_inputs[1],
            _bn3_inputs[0],_bn3_inputs[1],
            _relu3,
            _bn1_outputs[0],_bn1_outputs[1],
            _bn2_outputs[0],_bn2_outputs[1],
            _bn3_outputs[0],_bn3_outputs[1]]

target = tvm.target.Target("cuda")

def resnet_block():
    n, c, h, w = 64, 512, 56, 56

    task = auto_scheduler.SearchTask(func = tvm_resnet_block, args = (n, c, h, w), target = target)
    return task

task = resnet_block()
print(task.compute_dag)

log_file = "des.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(n_parallel=8,timeout=1000,min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=16,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

# Kill the measurement process
del measure_ctx

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

mod = tvm.build(sch, args, target)

for sub_mod in mod.imported_modules:
     print(sub_mod.get_source("cu"))

def run_batch_normalization(mod):
    n, c, h, w = 64, 512, 56, 56
    _input = np.random.uniform(size=(n, c, h, w)).astype(np.float32)

    _filter1 = np.random.uniform(size=(128, c, 1, 1)).astype(np.float32)
    _mean1 = np.random.uniform(size=(1, 128)).astype(np.float32)
    _var1 = np.random.uniform(size=(1, 128)).astype(np.float32)
    _mean11 = np.zeros(shape=(1, 128), dtype=np.float32)
    _var11 = np.zeros(shape=(1, 128), dtype=np.float32)

    _filter2 = np.random.uniform(size=(128, 128, 3 ,3)).astype(np.float32)
    _mean2 = np.random.uniform(size=(1, 128)).astype(np.float32)
    _var2 = np.random.uniform(size=(1, 128)).astype(np.float32)
    _mean22 = np.zeros(shape=(1, 128), dtype=np.float32)
    _var22 = np.zeros(shape=(1, 128), dtype=np.float32)

    _filter3 = np.random.uniform(size=(c, 128, 1, 1)).astype(np.float32)
    _mean3 = np.random.uniform(size=(1, c)).astype(np.float32)
    _var3 = np.random.uniform(size=(1, c)).astype(np.float32)
    _mean33 = np.zeros(shape=(1, c), dtype=np.float32)
    _var33 = np.zeros(shape=(1, c), dtype=np.float32)

    O = np.zeros(shape=(n, c, h, w), dtype=np.float32)

    dev = tvm.gpu()

    input_ = tvm.nd.array(_input, device=dev)

    filter1_ = tvm.nd.array(_filter1, device=dev)
    mean1_ = tvm.nd.array(_mean1, device=dev)
    var1_ = tvm.nd.array(_var1, device=dev)
    mean11_ = tvm.nd.array(_mean11, device=dev)
    var11_ = tvm.nd.array(_var11, device=dev)

    filter2_ = tvm.nd.array(_filter2, device=dev)
    mean2_ = tvm.nd.array(_mean2, device=dev)
    var2_ = tvm.nd.array(_var2, device=dev)
    mean22_ = tvm.nd.array(_mean22, device=dev)
    var22_ = tvm.nd.array(_var22, device=dev)

    filter3_ = tvm.nd.array(_filter3, device=dev)
    mean3_ = tvm.nd.array(_mean3, device=dev)
    var3_ = tvm.nd.array(_var3, device=dev)
    mean33_ = tvm.nd.array(_mean33, device=dev)
    var33_ = tvm.nd.array(_var33, device=dev)

    tvm_O = tvm.nd.empty(O.shape, device=dev)

    for i in range(10):
        mod(input_,
            filter1_, filter2_, filter3_, 
            mean1_, var1_, mean2_, var2_, mean3_, var3_,
            tvm_O,
            mean11_, var11_, mean22_, var22_, mean33_, var33_)