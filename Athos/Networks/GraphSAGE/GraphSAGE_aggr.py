# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import tempfile

import numpy as np

import tensorflow as tf
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "TFCompiler"))
import DumpTFMtData

FLAGS = None


def graphsage(input):
    """graphsage builds the graph for a net for .

    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.


    with tf.name_scope("fc"):
        W_fc1 = weight_variable([args.feature_size, args.hidden_size])
        b_fc1 = bias_variable([args.hidden_size])
        h_fc1 = tf.nn.relu(tf.matmul(input, W_fc1) + b_fc1)

    for i in range(args.num_layers):
        with tf.name_scope(f"aggr_{i}"):
            if args.aggr_method == "max":
                h_pool1 = max_pool_2x2(h_conv1)
            elif args.aggr_method == "max":
                h_pool1 = max_pool_2x2(h_conv1)


        for j in range(args.num_fc_per_layer):
            with tf.name_scope(f"fc_{i}_{j}"):
                output = args.hidden_size
                if i == args.num_fc_per_layer - 1:
                    output_size = args.embedding_size
                W_fc = weight_variable([256, output_size])
                b_fc = bias_variable([output_size])
                h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

    return (
        y_conv,
        keep_prob,
        [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2],
    )


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.constant(0.25, shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.25, shape=shape)
    return tf.Variable(initial)


def findLabel(oneHotAns):
    for i in range(10):
        if oneHotAns[i] == 1.0:
            return i
    return -1


def main(_):
    x = tf.ones(tf.float32, [FLAGS.batch_size, FLAGS.feature_size])
    y_conv, keep_prob, modelWeights = graphsage(x)
    pred = tf.argmax(y_conv, 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        output_tensor = None
        gg = tf.get_default_graph()
        for node in gg.as_graph_def().node:
            # if node.name == 'fc2/add':
            if node.name == "ArgMax":
                output_tensor = gg.get_operation_by_name(node.name).outputs[0]
        optimized_graph_def = DumpTFMtData.save_graph_metadata(
            output_tensor, sess, feed_dict
        )

        saver = tf.train.Saver(modelWeights)
        saver.restore(sess, "./TrainedModel/lenetSmallModel")

        start_time = time.time()
        prediction = sess.run([y_conv, keep_prob], feed_dict=feed_dict)
        duration = time.time() - start_time

        print("Duration of execution : ", duration)
        print("Result ::::::::: \n", prediction[0])

        print("Prediction: ", np.argmax(prediction[0]))
        print("Actual label: ", findLabel(imagey[0]))

        trainVarsName = []
        for node in optimized_graph_def.node:
            if node.op == "VariableV2":
                trainVarsName.append(node.name)
        trainVars = list(
            map(
                lambda x: tf.get_default_graph().get_operation_by_name(x).outputs[0],
                trainVarsName,
            )
        )
        DumpTFMtData.dumpImgAndWeightsDataSeparate(
            sess,
            imagex[0],
            trainVars,
            "LenetSmall_img_{0}.inp".format(curImageNum),
            "LenetSmall_weights_{0}.inp".format(curImageNum),
            15,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_neighbors",
        type=list,
        default=[10,10,10],
        help="A list of the numbers of neighbors of every layer",
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=256,
        help="The size of input feature",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The size of mini-batch",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="The size of the output of hidden layers",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=256,
        help="The size of the embedding",
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
