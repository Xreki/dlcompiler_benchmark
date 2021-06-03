#python3
import time
import os,sys
import numpy as np
import tensorflow as tf

#os.environ['XLA_FLAGS'] = '--xla_hlo_profile --xla_dump_to=./tmp/generated'
os.environ['TF_CPP_VMODULE'] = 'xla_compilation_cache=1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

shape = (224,224,3)
num_classes = 1000

tf.config.optimizer.set_jit("enabled")

resnet50 = tf.keras.applications.ResNet50(input_shape=shape, pooling="avg", classes=num_classes)
resnet50.compile(optimizer="SGD", loss="SparseCategoricalCrossentropy")

X = np.random.rand(256, 224, 224, 3)
Y = np.random.randint(0, 1000, size = (256))

#warm up
with tf.device("GPU:0"):
    resnet50.predict(x=X, batch_size=32, steps=8, workers=8)

time_start=time.time()
with tf.device("GPU:0"):
    resnet50.predict(x=X, batch_size=32, steps=8, workers=8)
    #resnet50.fit(x = X, y = Y, batch_size = 16, epochs = 4)
time_end=time.time()
print('time cost',time_end-time_start,'s')
