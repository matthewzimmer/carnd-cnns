"""
LeNet Architecture

HINTS for layers:

    Convolutional layers:

    tf.nn.conv2d
    tf.nn.max_pool

    For preparing the convolutional layer output for the
    fully connected layers.

    tf.contrib.flatten
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten


# NOTE: Feel free to change these.
EPOCHS = 10
BATCH_SIZE = 64

# >>> LAYER WIDTHS

# This is where the width of each layer gets set.

n_classes = 10
layer_width = {
    'layer_1': 6,
    'layer_2': 16,
    'flattened': 5*5*16,
    'fc1': 120,
    'fc2': n_classes
}

# Weights and Biases
# We'll use the layer widths defined above to create the weights and biases.

weights = {
    'layer_1': tf.Variable(tf.truncated_normal([5, 5, 1, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal([5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'fc1': tf.Variable(tf.truncated_normal([layer_width['flattened'], layer_width['fc1']])),
    'fc2': tf.Variable(tf.truncated_normal([layer_width['fc1'], n_classes]))
}

biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'fc1': tf.Variable(tf.zeros(layer_width['fc1'])),
    'fc2': tf.Variable(tf.zeros(n_classes))
}

def conv2d(x, W, b, strides=1, padding='VALID'):
    """
    Conv2D wrapper, with bias and relu activation.

    The tf.nn.conv2d() function computes the convolution against weight W as shown above.

    To make life easier, the code is using tf.nn.bias_add() to add the bias. Using tf.add() doesn't work when the
    tensors aren't the same shape.


    :param x: The input `Tensor`
    :param W: The weight `Tensor`
    :param b: The bias `Tensor`

    :param strides:
        In TensorFlow, stride is an array of 4 elements; the first element in the stride array indicates the stride for
        batch and last element indicates stride for feature/channel. It's good practice to remove the images or features
        you want to skip from the dataset than to use stride, so you can set the first and last element in the stride
        array to 1.

        The middle two elements are the strides for height and width respectively. I've mentioned stride as one number
        because you usually have a square stride where height = width. When someone says they are using a stride of 3,
        they usually mean tf.nn.conv2d(x, W, strides=[1, 3, 3, 1]).

    :return: A Rectified Linear Unit layer (ReLU) activation op for a SAME padding convolutional layer with
             specified strides applied.
    """

    # https://www.tensorflow.org/api_docs/python/nn.html#conv2d
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)

    # https://www.tensorflow.org/api_docs/python/nn.html#bias_add
    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)


# >>> MAX POOLING

def maxpool2d(x, k=2, padding='SAME'):
    """
    The tf.nn.max_pool() function does exactly what you would expect, it performs max pooling with the ksize parameter
    as the size of the filter.

    :param x:
    :param k: The size of the filter to be applied to the pooling algorithm.
    :return: Max pooling operation with a filter size `k` and a stride of `k`. Why stride k????
    """

    # https://www.tensorflow.org/api_docs/python/nn.html#max_pool
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)

# LeNet architecture:
# INPUT -> CONV -> ACT -> POOL -> CONV -> ACT -> POOL -> FLATTEN -> FC -> ACT -> FC
#
# Don't worry about anything else in the file too much, all you have to do is
# create the LeNet and return the result of the last fully connected layer.
def LeNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 28, 28, 1))
    # Squish values from 0-255 to 0-1.
    x /= 255.
    # Resize to 32x32.
    # We do this because LeNet's network trained on 32x32 images.
    x = tf.image.resize_images(x, (32, 32))

    # Layer 1 - 32*32*1 to 28*28*6
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1, k=2)  # to go from 28x28 to 14x14

    # Layer 2 - 14*14*6 to 10*10*16
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2, k=2)  # to go from 10x10 to 5x5

    # Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is
    # by using tf.contrib.layers.flatten, which is already imported for you.
    flattened = tf.contrib.layers.flatten(conv2)

    # Fully connected layer
    fc1 = tf.add(tf.matmul(flattened, weights['fc1']), biases['fc1'])
    fc1 = tf.nn.tanh(fc1) # activation

    # Output Layer - class prediction - 512 to 10
    fc2 = tf.add(tf.matmul(fc1, weights['fc2']), biases['fc2'])

    # TODO: Define the LeNet architecture.
    # Return the result of the last fully connected layer.
    return fc2


# MNIST consists of 28x28x1, grayscale images.
x = tf.placeholder(tf.float32, (None, 784))
# Classify over 10 digits 0-9.
y = tf.placeholder(tf.float32, (None, 10))
# Create the LeNet.
fc2 = LeNet(x)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def eval_data(dataset):
    """
    Given a dataset as input returns the loss and accuracy.
    """
    steps_per_epoch = dataset.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    total_acc, total_loss = 0, 0
    for step in range(steps_per_epoch):
        batch_x, batch_y = dataset.next_batch(BATCH_SIZE)
        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={x: batch_x, y: batch_y})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples


if __name__ == '__main__':
    # Load data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        steps_per_epoch = mnist.train.num_examples // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE

        # Train model
        for i in range(EPOCHS):
            for step in range(steps_per_epoch):
                batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
                loss = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})

            val_loss, val_acc = eval_data(mnist.validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation loss = {}".format(val_loss))
            print("Validation accuracy = {}".format(val_acc))

        # Evaluate on the test data
        test_loss, test_acc = eval_data(mnist.test)
        print("Test loss = {}".format(test_loss))
        print("Test accuracy = {}".format(test_acc))
