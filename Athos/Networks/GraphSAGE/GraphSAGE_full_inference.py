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


def graphsage(input, neigh_inputs):
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


    with tf.name_scope("fc_0"):
        W_fc = weight_variable([FLAGS.feature_size, FLAGS.hidden_size])
        b_fc = bias_variable([FLAGS.hidden_size])
        h = tf.nn.relu(tf.matmul(input, W_fc) + b_fc)

    for i in range(FLAGS.num_layers):
        with tf.name_scope(f"aggr_{i}"):
            if FLAGS.aggr_method == "SUM":
                for k in range(neigh_inputs[i].shape[0]):
                    h = h + neigh_inputs[i][k]
            elif FLAGS.aggr_method == "AVG_opt":
                for k in range(neigh_inputs[i].shape[0]):
                    h = h + neigh_inputs[i][k]
                r = bias_variable([1])
                h = h * r
            elif FLAGS.aggr_method == "AVG":
                h_and_neigh = tf.concat([neigh_inputs[i], tf.expand_dims(h, 0)], axis=0)
                h = tf.reduce_mean(h_and_neigh, axis=0)
            elif FLAGS.aggr_method == "EDGE":
                h_and_neigh = tf.concat([neigh_inputs[i], tf.expand_dims(h, 0)], axis=0)
                edge = weight_variable([FLAGS.num_neighbors[i] + 1, FLAGS.hidden_size])
                h = tf.concat([h_and_neigh, edge], axis=1)
                W_fc = weight_variable([FLAGS.hidden_size * 2, FLAGS.hidden_size])
                b_fc = bias_variable([FLAGS.hidden_size])
                h = tf.nn.relu(tf.matmul(h, W_fc) + b_fc)
            elif FLAGS.aggr_method == "MAX":
                h_and_neigh = tf.concat([neigh_inputs[i], tf.expand_dims(h, 0)], axis=0)
                h = tf.reduce_max(h_and_neigh, axis=0)
            elif FLAGS.aggr_method == "ATT":
                h_and_neigh = tf.concat([neigh_inputs[i], tf.expand_dims(h, 0)], axis=0)
                prev_h = weight_variable([FLAGS.num_neighbors[i] + 1, FLAGS.hidden_size])
                tmp_h_and_neigh = tf.concat([h_and_neigh, prev_h], axis=1)
                W_fc = weight_variable([FLAGS.hidden_size * 2, 1])
                b_fc = bias_variable([1])
                a = tf.nn.relu(tf.matmul(tmp_h_and_neigh, W_fc) + b_fc)
                a = tf.reshape(a, [-1])
                ea = tf.exp(a)
                ea = ea / tf.reduce_sum(ea)
                tmp_h = ea[:, None] * h_and_neigh  # Broadcasting multiplication
                h = tf.reduce_sum(tmp_h, axis=0)

        for j in range(FLAGS.num_fc_per_layer):
            with tf.name_scope(f"fc_{i}_{j}"):
                output = FLAGS.hidden_size
                if i == FLAGS.num_fc_per_layer - 1:
                    output_size = FLAGS.embedding_size
                W_fc = weight_variable([256, output_size])
                b_fc = bias_variable([output_size])
                h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc) + b_fc)

        if FLAGS.workload == "embedding":
            res = h_fc
        elif FLAGS.workload == "score":
            candidates = weight_variable([FLAGS.embedding_size, FLAGS.num_candidates])
            res = tf.matmul(h_fc, candidates)
        # elif FLAGS.workload == "rank":
        #     candidates = weight_variable([FLAGS.embedding_size, FLAGS.num_candidates])
        #     scores = tf.matmul(h_fc, candidates)
        #     res = topK(scores, 5)
    return res


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.constant(0.25, shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.25, shape=shape)
    return tf.Variable(initial)


def main(_):
    input = tf.constant(0.25, shape=[FLAGS.feature_size])
    neigh_inputs = []
    for i in range(len(FLAGS.num_neighbors)):
        neigh_inputs.append(tf.ones(tf.float32, [FLAGS.num_neighbors[i], FLAGS.hidden_size]))
    res = graphsage(input, neigh_inputs)

    with tf.Session() as sess:

        gg = tf.get_default_graph()
        optimized_graph_def = DumpTFMtData.save_graph_metadata(
            res, sess
        )

        start_time = time.time()
        prediction = sess.run(res)
        duration = time.time() - start_time

        print("Duration of execution : ", duration)
        print("Result ::::::::: \n", prediction[0])

        print("Prediction: ", np.argmax(prediction[0]))


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
    parser.add_argument(
        "--aggr_method",
        type=str,
        default="SUM",
        help="The method of neighbor aggregation SUM|AVG|EDGE|MAX|ATT",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=128,
        help="The number of candidates",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="score",
        help="embedding|score",
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
