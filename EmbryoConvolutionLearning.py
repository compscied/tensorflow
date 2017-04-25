#Original code from google MNIST tutorial modified for Embryos by DPS2018 Team 2
from datetime import datetime

# Load embryo images instead of mnist
from mnistembryo import read_data_sets
embryos = read_data_sets('', one_hot=True)
image_width = 64
image_height = 64
categories = 4

image_size = image_width*image_height # 64x64=4096

import tensorflow as tf

config = tf.ConfigProto(
#  device_count={'GPU': 0}, # disable GPU
  log_device_placement=True
  )

sess = tf.InteractiveSession(config=config)

x = tf.placeholder(tf.float32, shape=[None, image_size], name='InputImage')
y_ = tf.placeholder(tf.float32, shape=[None, categories], name='OutputLabels')

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1, name=name)
  return tf.Variable(initial)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape, name=name)
  return tf.Variable(initial)

def conv2d(x, W, name):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)

def max_pool_2x2(x, name):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)

W_conv1 = weight_variable([5, 5, 1, 32], name='Conv1w')
b_conv1 = bias_variable([32], name='Conv1b')

x_image = tf.reshape(x, [-1,image_height,image_width,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, name='Conv1') + b_conv1, name='Conv1ReLu')
h_pool1 = max_pool_2x2(h_conv1, name='Conv1Pool')

W_conv2 = weight_variable([5, 5, 32, 64], name='Conv2w')
b_conv2 = bias_variable([64], name='Conv2b')

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, name='Conv2') + b_conv2, name='Conv2ReLu')
h_pool2 = max_pool_2x2(h_conv2, name='Conv2Pool')

W_fc1 = weight_variable([16*16*64, 1024], name='FC1')
b_fc1 = bias_variable([1024], name='FC1Bias')

h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, categories], name='FC2')
b_fc2 = bias_variable([categories], name='FC2Bias')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

# Display current graph in tensorboard
summary_writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())

start_time=datetime.now()

for i in range(100000):
#for i in range(1000):
  batch = embryos.train.next_batch(480)
  if i%100 == 0:

    end_time = datetime.now()
    time_delta = end_time - start_time

    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})

    test_batch = embryos.test.next_batch(3000)

    test_accuracy = accuracy.eval(feed_dict={
      x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})

    print("%s: step %d, training accuracy %g, test accuracy %g"%(time_delta, i, train_accuracy, test_accuracy))

    end_time = datetime.now()
    time_delta = end_time - start_time
    print(time_delta)


  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

