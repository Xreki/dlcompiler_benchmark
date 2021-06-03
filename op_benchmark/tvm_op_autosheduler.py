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

@auto_scheduler.register_workload
def tvm_batch_normalization(n, c, h, w):
    _input = te.placeholder((n, c, h, w), name = 'A')
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

    _moving_mean = te.placeholder((1, c), name = 'mean')
    _moving_var = te.placeholder((1, c), name = 'var')
    _moving_mean_update = _moving_mean * 0.9 + _input_mean * 0.1
    _moving_var_update = _moving_var * 0.9 + _input_var * 0.1

    return [_input, _moving_mean, _moving_var, _output, _moving_mean_update, _moving_var_update]

@auto_scheduler.register_workload
def tvm_normalization(c, h, w, axis):
    A = te.placeholder((c,h,w), name = 'A')
    B = topi.sum(A, axis=axis, keepdims=True)
    C = B / c
    D = topi.subtract(A, C)
    E = topi.multiply(D, D)
    F = topi.sum(E, axis=axis, keepdims=True)
    G = F / c
    H = topi.sqrt(G)
    I = topi.divide(D, H)

    return [A,I]

@auto_scheduler.register_workload
def tvm_reduce(c, h, w, axis):
    A = te.placeholder((c, h ,w), name = 'A')
    B = topi.sum(A, axis=axis, keepdims=True)
    if axis == 0:
        C = B / c
    elif axis == 1:
        C = B / h
    else:
        C = B / w

    return [A,C]

@auto_scheduler.register_workload
def tvm_fuse_elementwise(c1, h1, w1, c2, h2, w2, c3, h3, w3, c4, h4, w4):
    A = te.placeholder((c1, h1, w1), name = 'A')
    B = te.placeholder((c2, h2, w2), name = 'B')
    C = te.placeholder((c3, h3, w3), name = 'C')
    D = te.placeholder((c4, h4, w4), name = 'D')

    E = topi.add(A, B)
    F = topi.subtract(E, C)
    G = topi.multiply(F, D)

    return [A,B,C,D,G]

@auto_scheduler.register_workload
def tvm_fuse_elementwise_no_broadcast(c, h, w):
    A = te.placeholder((c, h, w), name = 'A')
    B = te.placeholder((c, h, w), name = 'B')
    C = te.placeholder((c, h, w), name = 'C')
    D = te.placeholder((c, h, w), name = 'D')

    E = topi.add(A, B)
    F = topi.subtract(E, C)
    G = topi.multiply(F, D)

    return [A,B,C,D,G]

target = tvm.target.Target("cuda")

def batch_normalization():
    n, c, h, w = 128, 256, 32, 32
    task = auto_scheduler.SearchTask(func = tvm_batch_normalization, args = (n, c, h, w), target = target)
    return task

def normalization(axis):
    if(axis > 2):
        sys.exit("axis should be less than 3")
    c,h,w = 128,1024,256
    task = auto_scheduler.SearchTask(func = tvm_normalization, args = (c,h,w,axis), target = target)
    return task

def reduce(axis):
    if(axis > 2):
        sys.exit("axis should be less than 3")
    c, h, w = 128, 1024, 256
    task = auto_scheduler.SearchTask(func = tvm_reduce, args = (c, h, w, axis), target = target)
    return task

def element_wise():
    c1,h1,w1 = 256, 512, 1024
    c2,h2,w2 = 1, 512, 1024
    c3,h3,w3 = 256, 1, 1024
    c4,h4,w4 = 256, 512, 1

    task = auto_scheduler.SearchTask(func = tvm_fuse_elementwise, args = (c1,h1,w1,c2,h2,w2,c3,h3,w3,c4,h4,w4), target = target)
    return task

def element_wise_no_broadcast():
    c,h,w = 256, 512, 1024

    task = auto_scheduler.SearchTask(func = tvm_fuse_elementwise_no_broadcast, args = (c,h,w), target = target)
    return task

if pargs.op == "normalization":
    task = normalization(pargs.axis)
elif pargs.op == "reduce":
    task = reduce(pargs.axis)
elif pargs.op == "element_wise":
    task = element_wise()
elif pargs.op == "element_wise_no_broadcast":
    task = element_wise_no_broadcast()
elif pargs.op == "batch_normalization":
    task = batch_normalization()
else:
    sys.exit("op is unkown!")

print(task.compute_dag)

log_file = "des.json"
measure_ctx = auto_scheduler.LocalRPCMeasureContext(n_parallel=4,timeout=30,min_repeat_ms=300)
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
#mod.save("normalize.o")

#mod = tvm.load_module("normalize.tvm");
# Check correctness

def run_normalization(mod):
    c, h, w = 128, 1024, 256
    A = np.random.uniform(size=(c, h, w)).astype(np.float32)
    O = np.zeros(shape=(c, h, w), dtype=np.float32)

    dev = tvm.gpu()
    tvm_A = tvm.nd.array(A, device=dev)
    tvm_O = tvm.nd.empty(O.shape, device=dev)
    for i in range(10):
        mod(tvm_A, tvm_O)

def run_batch_normalization(mod):
    n, c, h, w = 128, 256, 32, 32
    A = np.random.uniform(size=(n, c, h, w)).astype(np.float32)
    mean = np.random.uniform(size=(1, c)).astype(np.float32)
    var = np.random.uniform(size=(1, c)).astype(np.float32)

    O = np.zeros(shape=(n, c, h, w), dtype=np.float32)
    m = np.zeros(shape=(1, c), dtype=np.float32)
    v = np.zeros(shape=(1, c), dtype=np.float32)

    dev = tvm.gpu()
    tvm_A = tvm.nd.array(A, device=dev)
    tvm_mean = tvm.nd.array(mean, device=dev)
    tvm_var = tvm.nd.array(var, device=dev)

    tvm_O = tvm.nd.empty(O.shape, device=dev)
    tvm_m = tvm.nd.empty(m.shape, device=dev)
    tvm_v = tvm.nd.empty(v.shape, device=dev)

    for i in range(10):
        mod(tvm_A, tvm_mean, tvm_var, tvm_O, tvm_m, tvm_v)

def run_element_wise(mod):
    c1,h1,w1 = 256, 512, 1024
    c2,h2,w2 = 1,   512, 1024
    c3,h3,w3 = 256, 1  , 1024
    c4,h4,w4 = 256, 512, 1

    A = np.random.uniform(size=(c1, h1, w1)).astype(np.float32)
    B = np.random.uniform(size=(c2, h2, w2)).astype(np.float32)
    C = np.random.uniform(size=(c3, h3, w3)).astype(np.float32)
    D = np.random.uniform(size=(c4, h4, w4)).astype(np.float32)
    O = np.zeros(shape=(c1, h1, w1), dtype=np.float32)

    dev = tvm.gpu()
    tvm_A = tvm.nd.array(A, device=dev)
    tvm_B = tvm.nd.array(B, device=dev)
    tvm_C = tvm.nd.array(C, device=dev)
    tvm_D = tvm.nd.array(D, device=dev)
    tvm_O = tvm.nd.empty(O.shape, device=dev)

    for i in range(10):
        mod(tvm_A, tvm_B, tvm_C, tvm_D, tvm_O)

def run_element_wise_no_broadcast(mod):
    c,h,w = 256, 512, 1024

    A = np.random.uniform(size=(c, h, w)).astype(np.float32)
    B = np.random.uniform(size=(c, h, w)).astype(np.float32)
    C = np.random.uniform(size=(c, h, w)).astype(np.float32)
    D = np.random.uniform(size=(c, h, w)).astype(np.float32)
    O = np.zeros(shape=(c, h, w), dtype=np.float32)

    dev = tvm.gpu()
    tvm_A = tvm.nd.array(A, device=dev)
    tvm_B = tvm.nd.array(B, device=dev)
    tvm_C = tvm.nd.array(C, device=dev)
    tvm_D = tvm.nd.array(D, device=dev)
    tvm_O = tvm.nd.empty(O.shape, device=dev)

    for i in range(10):
        mod(tvm_A, tvm_B, tvm_C, tvm_D, tvm_O)

def run_reduce(mod, axis):
    c, h, w = 128, 1024, 256
    A = np.random.uniform(size=(c, h, w)).astype(np.float32)
    if axis == 0:
        O = np.zeros(shape=(1, h, w), dtype=np.float32)
    elif axis == 1:
        O = np.zeros(shape=(c, 1, w), dtype=np.float32)
    elif axis == 2:
        O = np.zeros(shape=(c, h, 1), dtype=np.float32)

    dev = tvm.gpu()
    tvm_A = tvm.nd.array(A, device=dev)
    tvm_O = tvm.nd.empty(O.shape, device=dev)
    for i in range(10):
        mod(tvm_A, tvm_O)

if pargs.op == "batch_normalization":
    run_batch_normalization(mod)
elif pargs.op == "normalization":
    run_normalization(mod)
elif pargs.op == "reduce":
    run_reduce(mod, pargs.axis)
elif pargs.op == "element_wise":
    run_element_wise(mod)
elif pargs.op == "element_wise_no_broadcast":
    run_element_wise_no_broadcast(mod)

#run_element_wise(mod)
#A = te.placeholder((B,N), name = 'A')
#Alpha = te.placeholder((N,), name = 'Alpha')
#Beta = te.placeholder((N,), name = "Beta")

#C = te.compute((B,N), lambda x,y: A[x,y] + Alpha[y], name='C')
#D = te.compute((B,N), lambda x,y: C[x,y] * Beta[y], name='D')

#S = te.create_schedule(D.op)
#S[C].compute_inline()

#S[D].bind(D.op.axis[0], te.thread_axis("blockIdx.x"))
#S[D].bind(D.op.axis[1], te.thread_axis("threadIdx.x"))

#ir_mod = tvm.lower(S, [A, Alpha, Beta, C, D], simple_mode=True,name='x')
#mod = tvm.build(S, [A, Alpha, Beta, C, D], target='cuda', target_host="c", name='xx')

#print(ir_mod.astext(show_meta_data=True))
#print(mod.get_source())

for sub_mod in mod.imported_modules:
     print(sub_mod.get_source("cu"))
# print tir
#print("tir:\n", ir_m.astext(show_meta_data=True))
# print source code
#print("source code:\n",rt_m.get_source())
