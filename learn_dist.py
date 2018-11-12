# TODO(haija): Add license info.

import collections
import random

from absl import app
from absl import flags
import numpy
import tensorflow as tf

import model_base

flags.DEFINE_integer('batch_size', 32, 'Training batch size.')
flags.DEFINE_integer('num_iterations', 10, '')
flags.DEFINE_float('learn_rate', 0.01, 'SGD learning rate.')
flags.DEFINE_float('q_learn_rate', 0.001, 'SGD learning rate for Q.')
flags.DEFINE_string('model_class', 'MNISTModel',
                    'Must be a class name that inherits model_base.ResNet.')
flags.DEFINE_string('dataset_fn', 'GetMNIST',
                    'Must be a function name that returns: '
                    '(train_x, train_y), (test_x, test_y).')

VALIDATION_CLASS_WEIGHTS = {
    7: 4,
    8: 4,
    3: 4,

    0: 1,
    1: 1,

    2: 0,   # Skip
    4: 0,   # Skip
    5: 0,   # Skip
    6: 0,   # Skip
    9: 0,   # Skip
}
COLORS = [
    '#ff0000',
    '#0000ff',
    '#00ff00',
    '#ff00ff',
    '#00ffff',
    '#ffff00',
    '#000000',
    '#7f007f',
    '#007f7f',
    '#7f7f00',
]
COLORS = {
    7: '#ff0000',
    3: '#ffff00',
    8: '#ff00ff',
    1: '#0000ff',
    0: '#ff00ff',

    2: '#ff7f00',
    4: '#007f00',
    5: '#7f007f',
    6: '#00007f',
    9: '#7f0000'
}

FLAGS = flags.FLAGS

### Datasets
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.datasets import mnist

def GetCIFAR10():
  return cifar10.load_data()

def GetMNIST():
  (train_x, train_y), (test_x, test_y) = mnist.load_data()
  train_x = numpy.expand_dims(train_x, 3)
  train_y = numpy.expand_dims(train_y, 1)
  test_x = numpy.expand_dims(test_x, 3)
  test_y = numpy.expand_dims(test_y, 1)
  return (train_x[:2000], train_y[:2000]), (test_x[:1000], test_y[:1000])


def one_hot(y, num_classes=10):
  """Given a numpy matrix of shape [N, 1], ret one-hot matrix [N, num_classes]"""
  arr = 0.5*numpy.ones(shape=(y.shape[0], num_classes), dtype='float32')
  arr[range(y.shape[0]), y[:, 0]] = 1.0
  return arr


### Models
class MNISTModel(model_base.ResNet):

  def __init__(self, layer_parameters=None, Q=None):
    """

    Args:
      layer_parameters: If set, then it must be a tuple containing 2 lists, the
        first contains the layer_specs and the second must contain dictionaries
        if the parameter values. If set, then the weights (in the second tuple)
        will be used for the parameter values, and the layer_specs will be used
        to verify that the layer config is identical. If not used, the layers
        will be built with new parameters.
      Q: If given, must be a tensor or a placeholder, that will be used on the
        backward pass to compute gradients w.r.t. parameters.
    """
    self.is_training = tf.placeholder_with_default(True, [], name='is_training')
    self.num_classes = 10
    super(MNISTModel, self).__init__(self.is_training)
    if layer_parameters is not None:
      layer_specs, layer_weights = layer_parameters
      self.set_custom_weights(layer_specs, layer_weights)
    self.x = tf.placeholder(tf.float32, (None, 28, 28, 1))
    self.y = tf.placeholder(tf.float32, (None, self.num_classes))
    self.Q = Q
    self.logits = self.forward_pass(self.x)

  def forward_pass(self, x):
    x = x / 128 - 1
    x = self._conv(x, 1, 16, 1, pad=False)
    x = self._batch_norm(x)
    x = self._relu(x)
    x = self._conv(x, 5, 16, 1, pad=False)
    x = self._batch_norm(x)
    x = self._relu(x)
    x = self._avg_pool(x, 2, 2, padding='VALID')
    x = self._conv(x, 4, 16, 1, pad=False)
    x = self._batch_norm(x)
    x = self._relu(x)

    flattened_dim = x.shape[1] * x.shape[2] * x.shape[3]
    x = tf.reshape(x, [-1, flattened_dim])

    x = self._fully_connected(x, 100)
    x = self._batch_norm(x)
    x = self._relu(x)
    x = self._fully_connected(x, 10)

    return x


class Cifar10Resnet(model_base.ResNet):

  def __init__(self, layer_parameters=None, Q=None):
    """

    Args:
      layer_parameters: If set, then it must be a tuple containing 2 lists, the
        first contains the layer_specs and the second must contain dictionaries
        if the parameter values. If set, then the weights (in the second tuple)
        will be used for the parameter values, and the layer_specs will be used
        to verify that the layer config is identical. If not used, the layers
        will be built with new parameters.
      Q: If given, must be a tensor or a placeholder, that will be used on the
        backward pass to compute gradients w.r.t. parameters.
    """
    self.is_training = tf.placeholder_with_default(True, [], name='is_training')
    self.num_classes = 10
    super(Cifar10Resnet, self).__init__(self.is_training)
    if layer_parameters is not None:
      layer_specs, layer_weights = layer_parameters
      self.set_custom_weights(layer_specs, layer_weights)
    self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    self.y = tf.placeholder(tf.float32, (None, self.num_classes))

    num_layers = 44
    self.n = (num_layers - 2) // 6
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]
    self.Q = Q

    self.logits = self.forward_pass(self.x)


  # Copied from:
  # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_model.py
  def forward_pass(self, x):
    x = x / 128 - 1
    x = self._conv(x, 3, 16, 1)
    x = self._batch_norm(x)
    x = self._relu(x)

    # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
    res_func = self._residual_v1

    # 3 stages of block stacking.
    for i in range(3):
      with tf.name_scope('stage'):
        for j in range(self.n):
          if j == 0:
            # First block in a stage, filters and strides may change.
            x = res_func(x, 3, self.filters[i], self.filters[i + 1],
                         self.strides[i])
          else:
            # Following blocks in a stage, constant filters and unit stride.
            x = res_func(x, 3, self.filters[i + 1], self.filters[i + 1], 1)

    x = self._global_avg_pool(x)
    x = self._fully_connected(x, self.num_classes) 
    return x
    

### Data Manipulations
def shuffle(x, y):
  assert x.shape[0] == y.shape[0]
  permutation = list(range(x.shape[0]))
  random.shuffle(permutation)
  return x[permutation], y[permutation]


class DatasetSampler(object):
  
  def __init__(self, train_x, train_y, test_x, test_y, percent_validate=0.2,
               class_weights=None):
    train_x, train_y = shuffle(train_x, train_y)
    num_train = int((1-percent_validate) * train_x.shape[0])
    self.train_x = train_x[:num_train]
    self.train_y = train_y[:num_train]
    self.class_weights = class_weights or VALIDATION_CLASS_WEIGHTS
    self.validate_x, self.validate_y = self._sample_with_class_weights(
        train_x[num_train:], train_y[num_train:])
    test_x, test_y = shuffle(test_x, test_y)
    self.test_x, self.test_y = self._sample_with_class_weights(test_x, test_y)
    self.train_q = numpy.ones([num_train], dtype='float32') * 0.5

  def _sample_with_class_weights(self, x, y):
    """Assumes that `x` and `y` are already shuffled."""
    actual_counts = collections.Counter(y[:, 0])
    # count over ratio
    sample_unit = min([actual_counts[k] / (float(v) or 1)  # remove / 0
                       for (k, v) in self.class_weights.iteritems()])
    num_samples = {k: int(v * sample_unit)
                   for (k, v) in self.class_weights.iteritems()}

    collected_x = []
    collected_y = []
    for xi, yi in zip(x, y):
      if num_samples.get(yi[0], 0) > 0:
        collected_x.append(xi)
        collected_y.append(yi)
        num_samples[yi[0]] -= 1 
    return numpy.array(collected_x), numpy.array(collected_y)

  def shuffle_train(self):
    permutation = list(range(self.train_x.shape[0]))
    random.shuffle(permutation)
    self.train_x = self.train_x[permutation]
    self.train_y = self.train_y[permutation]
    self.train_q = self.train_q[permutation]


def main(unused_argv):
  model_class = globals()[FLAGS.model_class]
  dataset_fn = globals()[FLAGS.dataset_fn]

  # Load dataset
  (train_x, train_y), (test_x, test_y) = dataset_fn()
  
  # Sampler (samples test+validate from same distribution)
  sampler = DatasetSampler(train_x, train_y, test_x, test_y)

  # Q.
  q_placeholder = tf.placeholder(tf.float32, [None], name='q')
  train_Q = tf.nn.softmax(q_placeholder)

  # Model
  model = model_class(Q=train_Q)


  classification_loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=model.logits, labels=model.y)

  q_weighted_loss = tf.transpose(tf.transpose(classification_loss) * train_Q)



  # weighted_loss = tf.reduce_sum(classification_loss)  # * train_Q

  # updated_params = 
  learn_rate = tf.placeholder(tf.float32, [], name='learn_rate')

  trainable_vars = tf.trainable_variables()
  layer_ys = [l.y for l in model.custom_layers]
  layer_xs = [l.x for l in model.custom_layers]
  layer_ys_grad_placeholders = [l.grad_wrt_y for l in model.custom_layers]

  all_grads = tf.gradients(q_weighted_loss, trainable_vars + layer_ys)
  trainable_grads, layer_ys_grads = (
      all_grads[:len(trainable_vars)], all_grads[len(trainable_vars):])

  layer_update_rules = {}
  layers_custom_weights = []
  assign_ops = []

  # train_optimizer = tf.train.AdamOptimizer(learn_rate)
  for i, l in enumerate(model.custom_layers):
    custom_weights = {}
    for name, var in l.weights.iteritems():
      #import IPython; IPython.embed()
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/training_ops.cc
      if var in trainable_vars:
        var_pos = trainable_vars.index(var)
        var_grad = trainable_grads[var_pos]
        train_step_op = var.assign(var - learn_rate * var_grad)  # SGD
        # train_optimizer.apply_gradients([(var_grad, var)])
        #next_theta_q_parametrized = train_optimizer.apply_gradients(
        #  [(l.grad_wrt_weight[name], var)])

        assign_ops.append(train_step_op)
        # import IPython; IPython.embed()
        update_rule = var - learn_rate * l.grad_wrt_weight[name]
        # assign_ops.append(var.assign(var - learn_rate * var_grad))
      layer_update_rules[var] = update_rule
      custom_weights[name] = update_rule
    layers_custom_weights.append(custom_weights)

  # Make another model sharing those params.
  validate_model = model_class(
      (model.custom_layer_specs, layers_custom_weights))

  validate_loss = tf.nn.softmax_cross_entropy_with_logits(
      logits=validate_model.logits, labels=validate_model.y)

  dlossv_dq = tf.gradients(validate_loss, q_placeholder, stop_gradients=layer_xs)
  
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  batch_validation_x = sampler.validate_x
  batch_validation_y = sampler.validate_y
  def step(i, lr=FLAGS.learn_rate, update_theta=True, update_q=True):
    sampler.shuffle_train()
    for b_start in xrange(0, len(sampler.train_x), FLAGS.batch_size):
      b_end = min(b_start + FLAGS.batch_size, len(sampler.train_x))
      num_examples = b_end - b_start

      if update_q:
        ### 
          # For a training batch (X, Y) we calculate the gradient of the loss w.r.t.
        # all layers in the neural network.
        forward_backward_pass_feed_dict = {
            model.x: sampler.train_x[b_start:b_end],
            model.y: one_hot(sampler.train_y[b_start:b_end]),
            q_placeholder: numpy.ones([num_examples]),
        }
        # backpass, forward pass
        #    |        |
        #    V        V
        v_ys_grads, v_layer_xs = sess.run(
            (layer_ys_grads, layer_xs), forward_backward_pass_feed_dict)

        # Get gradients wrt distribution Q.
        v_feed_dict = {
            q_placeholder: sampler.train_q[b_start:b_end],
            validate_model.x: batch_validation_x,
            validate_model.y: one_hot(batch_validation_y),
            learn_rate: lr,
        }

        ### Feed-in forward-backward pass above, but into Q-network.
        # Feed inputs to all the layers.
        for t, v in zip(layer_xs, v_layer_xs):
          v_feed_dict[t] = v
        # Feed gradients w.r.t. layer outputs.
        for t, v in zip(layer_ys_grad_placeholders, v_ys_grads):
          v_feed_dict[t] = v

        
        vdq, vlogits = sess.run((dlossv_dq, validate_model.logits),
                                v_feed_dict)

        # Update q (that parametrizes Q).
        sampler.train_q[b_start:b_end] -= (
             FLAGS.q_learn_rate * vdq[0]
             + 1e-4 * sampler.train_q[b_start:b_end])
        # with 1e-4 L2 regularization

      if update_theta:
        # Update model parameters, using the updated Q.
        sess.run(assign_ops, feed_dict={
          q_placeholder: sampler.train_q[b_start:b_end],
          model.x: sampler.train_x[b_start:b_end],
          model.y: one_hot(sampler.train_y[b_start:b_end]),
          learn_rate: lr,
        })
      # assess quality
      #accuracy = numpy.mean(vlogits.argmax(axis=1) == batch_validation_y[:, 0])
      #print accuracy

    # Test accuracy
    test_logits = sess.run(model.logits, {model.x: sampler.test_x})
    test_accuracy = numpy.mean(
        test_logits.argmax(axis=1) == sampler.test_y[:, 0])
    print 'Test accuracy = %f' % test_accuracy
    q_means = collections.Counter()
    q_counts = collections.Counter()
    for i in xrange(len(sampler.train_q)):
      q_means[sampler.train_y[i, 0]] += sampler.train_q[i]
      q_counts[sampler.train_y[i, 0]] += 1

    for i in q_means.keys():
      q_means[i] /= q_counts[i]
    print q_means
  
  import IPython; IPython.embed()
  """
for i in xrange(10):
    print '======== =====  Meta iteration %i' % i
    for j in xrange(20):
        print 'theta update %i.%i' % (i, j)
        step(i, lr=FLAGS.learn_rate*0.5, update_q=False, update_theta=True)
    for j in xrange(2):
        print 'Q update %i.%i' % (i, j)
        step(i, update_q=True, update_theta=False)
        
"""

  import matplotlib.pyplot as plt
  fig, ax = plt.subplots()

  sorted_q_y = sorted(zip(sampler.train_q, sampler.train_y[:, 0]))[::-1]
  plot_q, plot_class = zip(*sorted_q_y)
  plot_q = numpy.array(plot_q)
  plot_class = numpy.array(plot_class)
  bar_width = 1
  for i in [2,4,5,6,9]:
    positions = numpy.nonzero(plot_class == i)[0]
    color = COLORS[i]
    q_values = plot_q[positions]
    plt.bar(positions, q_values, bar_width, color=color, label=str(i), linewidth=0)

  plt.bar([len(plot_q)], [0], bar_width, color='b', linewidth=0)

  plt.legend()
  plt.show()

  return 0

if __name__ == '__main__':
  app.run(main)
