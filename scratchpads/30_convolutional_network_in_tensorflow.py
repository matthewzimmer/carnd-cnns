from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('.', one_hot=True, reshape=False)

import tensorflow as tf

# >>> PARAMETERS

learning_rate = 1e-3  # 0.001
batch_size = 128
training_epochs = 30
n_classes = 10  # MNIST total class (0-9 digits)

# >>> LAYER WIDTHS

# This is where the width of each layer gets set.

layer_width = {
    'layer_1': 32,
    'layer_2': 64,
    'layer_3': 128,
    'fully_connected': 512
}

# Weights and Biases
# We'll use the layer widths defined above to create the weights and biases.

weights = {
    'layer_1': tf.Variable(tf.truncated_normal([5, 5, 1, layer_width['layer_1']])),
    'layer_2': tf.Variable(tf.truncated_normal([5, 5, layer_width['layer_1'], layer_width['layer_2']])),
    'layer_3': tf.Variable(tf.truncated_normal([5, 5, layer_width['layer_2'], layer_width['layer_3']])),
    'fully_connected': tf.Variable(tf.truncated_normal([4 * 4 * 128, layer_width['fully_connected']])),
    'out': tf.Variable(tf.truncated_normal([layer_width['fully_connected'], n_classes]))
}

biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'layer_3': tf.Variable(tf.zeros(layer_width['layer_3'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(n_classes))
}


def conv2d(x, W, b, strides=1):
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
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')

    # https://www.tensorflow.org/api_docs/python/nn.html#bias_add
    x = tf.nn.bias_add(x, b)

    return tf.nn.relu(x)


# >>> MAX POOLING

def maxpool2d(x, k=2):
    """
    The tf.nn.max_pool() function does exactly what you would expect, it performs max pooling with the ksize parameter
    as the size of the filter.

    :param x:
    :param k: The size of the filter to be applied to the pooling algorithm.
    :return: Max pooling operation with a filter size `k` and a stride of `k`. Why stride k????
    """

    # https://www.tensorflow.org/api_docs/python/nn.html#max_pool
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases):
    # Layer 1 - 28*28*1 to 14*14*32
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1)

    # Layer 2 - 14*14*32 to 7*7*64
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2)

    # Layer 3 - 7*7*64 to 4*4*128
    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv3 = maxpool2d(conv3)

    # Fully connected layer - 4*4*128 to 512
    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['fully_connected'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['fully_connected']), biases['fully_connected'])

    fc1 = tf.nn.tanh(fc1)

    # Output Layer - class prediction - 512 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


# NOW LETS RUN IT
# tf Graph input
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])
logits = conv_net(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            # Display logs per epoch step
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
