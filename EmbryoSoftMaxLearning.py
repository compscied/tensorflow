#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#image_width = 28
#image_height = 28
#categories = 10

# Load embryo images instead of mnist
from mnistembryo import read_data_sets
mnist = read_data_sets('', one_hot=True)
image_width = 64
image_height = 64
categories = 4

image_size = image_width*image_height # 28x28=784, 64x64=4096

import tensorflow as tf
sess = tf.InteractiveSession()

batch_size = 100

x = tf.placeholder(tf.float32, shape=[None, image_size], name="InputImage")
y_ = tf.placeholder(tf.float32, shape=[None, categories], name="OutputLabel")

W = tf.Variable(tf.zeros([image_size,categories]))
b = tf.Variable(tf.zeros([categories]))

sess.run(tf.initialize_all_variables())

y = tf.matmul(x,W) + b

# Display current graph in tensorboard
summary_writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Test to view first image
#from PIL import Image
#batch = mnist.train.next_batch(batch_size)
#im = batch[0][0].reshape(image_height, image_width) * 255
#img = Image.fromarray(im)
#img.show()

for i in range(1000):
  batch = mnist.train.next_batch(batch_size)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Test to view weights
from PIL import Image
im = W.reshape(image_height, image_width)
img = Image.fromarray(im)
img.show()

