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

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE_W = 17
IMAGE_SIZE_H = 17
IMAGE_PIXELS = IMAGE_SIZE_W * IMAGE_SIZE_H


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
BATCH_SIZE = 1

def inference(images, hidden_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden_units: Array of sizes of the hidden layers.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  num_elements = len(hidden_units)
  i = 1

  prev_tensor = images

  while i < num_elements:
    #layer_name = 'Layer' + str(i)
    #with tf.name_scope(layer_name):
      weights = tf.Variable(
        tf.truncated_normal([hidden_units[i-1], hidden_units[i]],
                            stddev=1.0 / math.sqrt(float(hidden_units[i-1]))),
        name='weights')
      biases = tf.Variable(tf.zeros([hidden_units[i]]),
                         name='biases')
      if i == num_elements - 1:
          new_tensor = (tf.matmul(prev_tensor, weights) + biases)
      else:
          new_tensor = tf.nn.relu(tf.matmul(prev_tensor, weights) + biases)
          #new_tensor = tf.nn.sigmoid(tf.matmul(prev_tensor, weights) + biases)
      prev_tensor = new_tensor
      i = i + 1
  return prev_tensor


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """

  #  + 0.85* tf.nn.l2_loss(w) + 0.15* tf.reduce_mean(tf.abs(w))
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')



def training(loss, epoch_length, initial_learning_rate, num_epochs_per_decay, learning_rate_decay_factor):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Variables that affect learning rate.
  num_batches_per_epoch = epoch_length
  decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(initial_learning_rate,
                                  global_step,
                                  decay_steps,
                                  learning_rate_decay_factor,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(lr)
  # Create a variable to track the global step.
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, lr


def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))








def CIFAR_inference(images, num_classes):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  #with tf.variable_scope('conv1') as scope:

  tf_4d_reshape = tf.reshape(images, [1, 17, 17, 1])

  kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 50], stddev=1e-4), name='weights')
  conv = tf.nn.conv2d(tf_4d_reshape, kernel, [1, 1, 1, 1], padding='SAME')
  biases = tf.Variable(tf.zeros([50]), name='biases')

  bias = tf.nn.bias_add(conv, biases)
  conv1 = tf.nn.relu(bias, name='convolution_layer')

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pooling_layer')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='normalisation_layer')

  # local3
  with tf.variable_scope('fully_connected_layer') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    dim = 1
    for d in norm1.get_shape()[1:].as_list():
      dim *= d
    reshape = tf.reshape(norm1, [1, dim])

    weights = tf.Variable(
      tf.truncated_normal([dim, 200],
                          stddev=1.0 / math.sqrt(float(dim))),
      name='weights')
    biases = tf.Variable(tf.zeros([200]),
                         name='biases')

    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = tf.Variable(
      tf.truncated_normal([200, num_classes], stddev=1.0 / math.sqrt(float(200))), name='weights')
    biases = tf.Variable(tf.zeros([num_classes]),
                         name='biases')

    softmax_linear = (tf.matmul(local3, weights) + biases)

  return softmax_linear
