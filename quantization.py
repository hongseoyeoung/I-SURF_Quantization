import tensorflow as tf
from vgg13practice import cnn_model_fn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.logging.set_verbosity(tf.logging.INFO)

mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/mnist_vgg13_model")

mnist = tf.keras.datasets.mnist
(train_data, train_labels), (eval_data, eval_labels) = mnist.load_data()
train_data = train_data.reshape(60000, 28, 28, 1)
train_data = train_data / 255.0
train_data = np.asarray(train_data, dtype=np.float32)
train_labels = np.asarray(train_labels, dtype=np.int32)
eval_data = eval_data.reshape(10000, 28, 28, 1)
eval_data = eval_data / 255.0
eval_data = np.asarray(eval_data, dtype=np.float32)
eval_labels = np.asarray(eval_labels, dtype=np.int32)

mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="tmp/mnist_vgg13_model")
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=100, shuffle=True)
#mnist_classifier.train(input_fn=train_input_fn, steps=55000, hooks=None)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)