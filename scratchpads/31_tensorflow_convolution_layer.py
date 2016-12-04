"""
Setup the strides, padding and filter weight/bias such that
the output shape is (1, 2, 2, 3).
"""
import tensorflow as tf
import numpy as np

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)

# Input Size:    4x4x1
# Filter Size:   2x2x3
# Stride:        2x2

# SAME
# output_size = (W/S)
# 4/2 = 2


# VALID
#  out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
#  out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

# out_height = ceil(float(4 - 2 + 1) / float(2)) = ceil(1.5) = 2
# out_width  = ceil(float(4 - 2 + 1) / float(2)) = ceil(1.5) = 2

def conv2d(input):
    # Filter (weights and bias)
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth,)
    # TODO: Define the filter weights `F_W` and filter bias `F_b`.
    # NOTE: Remember to wrap them in `tf.Variable`, they are trainable parameters after all.
    F_W = tf.Variable(tf.truncated_normal([2, 2, 1, 3]))
    F_b = tf.Variable(tf.zeros(3))
    # TODO: Set the stride for each dimension (batch_size, height, width, depth)
    strides = [1, 2, 2, 1]
    # TODO: set the padding, either 'VALID' or 'SAME'.
    padding = 'SAME'

    # interestingly, 'VALID' padding works here once you read the documentation at
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#convolution
    # and realize the VALID height/width algorithm:
    #
    # For the 'VALID' padding, the output height and width are computed as:
    #
    #  out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    #  out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
    #
    #  and the padding values are always zero. The output is then computed as
    #
    #   output[b, i, j, :] = sum_{di, dj} input[b, strides[1] * i + di - pad_top, strides[2] * j + dj - pad_left, ...] * filter[di, dj, ...]

    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
    # `tf.nn.conv2d` does not include the bias computation so we have to add it ourselves after.
    return tf.nn.conv2d(input, F_W, strides, padding) + F_b

out = conv2d(X)

print(out)
print('conv2d tensor shape:', out.get_shape())