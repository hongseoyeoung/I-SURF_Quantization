import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from bit_wise_And import bit_wise_And, masking
from vgg13practice import cnn_model_fn
tf.logging.set_verbosity(tf.logging.INFO)

mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/mnist_vgg13_model")



dir = './weights/'
if not os.path.exists(dir) : 
    os.mkdir(dir)

for layer in mnist_classifier.get_variable_names():
    value = mnist_classifier.get_variable_value(layer).flatten()
    filepath = dir + layer.split('/')[0]+".txt"
    print(filepath)
    np.savetxt(filepath, value)

dir1 = './quantized_weights/'
if not os.path.exists(dir1) : 
    os.mkdir(dir1)

mask = masking(16)
for layer in mnist_classifier.get_variable_names():
    value = mnist_classifier.get_variable_value(layer).flatten()
    for num in range(len(value)):
        value[num] = bit_wise_And(value[num], mask)
    
    filepath = dir1 + layer.split('/')[0]+".txt"
    np.savetxt(filepath, value)