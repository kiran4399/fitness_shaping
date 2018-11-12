# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

# Copied from github:
# tensorflow/models/master/tutorials/image/cifar10_estimator/model_base.py

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""

import tensorflow as tf


def MakeLayerClass(base_layer_type):
  class CustomWeightLayer(base_layer_type):
    weights = None
    backprop = False

    def memorize_weight(self, name, tensor):
      if self.weights is None:
        self.weights = {}
      self.weights[name] = tensor

    def add_weight(self, name, shape, **kwargs):
      if self.weights is not None and name in self.weights:
        print 'utilizing memorized weight %s' % name
        return self.weights[name]

      weight = super(CustomWeightLayer, self).add_weight(name, shape, **kwargs)
      self.memorize_weight(name, weight)
      print 'Produced weight %s' % weight.name
      return weight

    def set_q_backprop(self, Q):
      """Sets the vector Q that is only used during the backprop."""
      self.backprop = True
      self.Q = Q

    def apply(self, x):
      self.y = super(CustomWeightLayer, self).apply(x)
      self.x = x
      if self.backprop:
        self.grad_wrt_y = tf.placeholder(tf.float32, self.y.shape)
        weighted_y = tf.transpose(tf.transpose(self.grad_wrt_y) * self.Q)
        # self.grad_wrt_x = tf.gradients(self.y, x, grad_ys=self.grad_wrt_y)
        
        # 
        names, weight_tensors = zip(*self.weights.items())
        grad_weights = tf.gradients(self.y, weight_tensors,
                                    grad_ys=weighted_y)
        self.grad_wrt_weight = dict(zip(names, grad_weights))
      return self.y

      
  return CustomWeightLayer


CustomWeightConv2d = MakeLayerClass(tf.layers.Conv2D)
CustomWeightDenseLayer = MakeLayerClass(tf.layers.Dense)


class ResNet(object):
  """ResNet model."""

  # Defaults copied from:
  # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_main.py 
  def __init__(self, is_training, batch_norm_decay=0.997,
               batch_norm_epsilon=1e-5,
               data_format='channels_last'):
    """ResNet constructor.

    Args:
      is_training: if build training or inference model.
      data_format: the data_format used during computation.
                   one of 'channels_first' or 'channels_last'.
    """
    self._batch_norm_decay = batch_norm_decay
    self._batch_norm_epsilon = batch_norm_epsilon
    self._is_training = is_training
    assert data_format in ('channels_first', 'channels_last')
    self._data_format = data_format
    self.custom_layers = []
    self.custom_layer_specs = []
    self._upcoming_custom_specs = self._upcoming_custom_weights = None

  def set_custom_weights(self, layers_specs, weights):
    self._upcoming_custom_specs = layers_specs
    self._upcoming_custom_weights = weights

  def forward_pass(self, x):
    raise NotImplementedError(
        'forward_pass() is implemented in ResNet sub classes')

  def _residual_v1(self,
                   x,
                   kernel_size,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers, using Plan A for shortcut connection."""

    del activate_before_residual
    with tf.name_scope('residual_v1') as name_scope:
      orig_x = x

      x = self._conv(x, kernel_size, out_filter, stride)
      x = self._batch_norm(x)
      x = self._relu(x)

      x = self._conv(x, kernel_size, out_filter, 1)
      x = self._batch_norm(x)

      if in_filter != out_filter:
        orig_x = self._avg_pool(orig_x, stride, stride)
        pad = (out_filter - in_filter) // 2
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = self._relu(tf.add(x, orig_x))

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _residual_v2(self,
                   x,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers with preactivation, plan A shortcut."""

    with tf.name_scope('residual_v2') as name_scope:
      if activate_before_residual:
        x = self._batch_norm(x)
        x = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x = self._batch_norm(x)
        x = self._relu(x)

      x = self._conv(x, 3, out_filter, stride)

      x = self._batch_norm(x)
      x = self._relu(x)
      x = self._conv(x, 3, out_filter, [1, 1, 1, 1])

      if in_filter != out_filter:
        pad = (out_filter - in_filter) // 2
        orig_x = self._avg_pool(orig_x, stride, stride)
        if self._data_format == 'channels_first':
          orig_x = tf.pad(orig_x, [[0, 0], [pad, pad], [0, 0], [0, 0]])
        else:
          orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad, pad]])

      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _bottleneck_residual_v2(self,
                              x,
                              in_filter,
                              out_filter,
                              stride,
                              activate_before_residual=False):
    """Bottleneck residual unit with 3 sub layers, plan B shortcut."""

    with tf.name_scope('bottle_residual_v2') as name_scope:
      if activate_before_residual:
        x = self._batch_norm(x)
        x = self._relu(x)
        orig_x = x
      else:
        orig_x = x
        x = self._batch_norm(x)
        x = self._relu(x)

      x = self._conv(x, 1, out_filter // 4, stride, is_atrous=True)

      x = self._batch_norm(x)
      x = self._relu(x)
      # pad when stride isn't unit
      x = self._conv(x, 3, out_filter // 4, 1, is_atrous=True)

      x = self._batch_norm(x)
      x = self._relu(x)
      x = self._conv(x, 1, out_filter, 1, is_atrous=True)

      if in_filter != out_filter:
        orig_x = self._conv(orig_x, 1, out_filter, stride, is_atrous=True)
      x = tf.add(x, orig_x)

      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _apply_custom_weight_layer(self, layer_in, layer_class, *args, **kwargs):
    layer = layer_class(*args, **kwargs)
    if self.Q is not None:
      layer.set_q_backprop(self.Q)

    spec = (tuple(layer_in.shape.as_list()), layer_class, tuple(args),
            tuple(sorted(kwargs.items())))
  
    if self._upcoming_custom_specs:
      assert spec == self._upcoming_custom_specs[0]
      layer_weights = self._upcoming_custom_weights[0]
      for k, v in layer_weights.iteritems():
        layer.memorize_weight(k, v)

      self._upcoming_custom_specs = self._upcoming_custom_specs[1:]
      self._upcoming_custom_weights = self._upcoming_custom_weights[1:]

    layer_output = layer.apply(layer_in)
    self.custom_layers.append(layer)

    self.custom_layer_specs.append(spec)
    return layer_output

  def _conv(self, x, kernel_size, filters, strides, pad=True, is_atrous=False):
    """Convolution."""

    if pad:
      padding = 'SAME'
      if not is_atrous and strides > 1:
        pad = kernel_size - 1
        pad_beg = pad // 2
        pad_end = pad - pad_beg
        if self._data_format == 'channels_first':
          x = tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
          x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        padding = 'VALID'
    else:
      padding = 'VALID'

    return self._apply_custom_weight_layer(
        x, CustomWeightConv2d,
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=self._data_format)

  def _batch_norm(self, x):
    if self._data_format == 'channels_first':
      data_format = 'NCHW'
    else:
      data_format = 'NHWC'
      # batch_weights
    return tf.contrib.layers.batch_norm(
        x,
        decay=self._batch_norm_decay,
        center=False,
        scale=False,  # TODO(haija): This and above were True
        epsilon=self._batch_norm_epsilon,
        is_training=self._is_training,
        fused=True,
        data_format=data_format)

  def _relu(self, x):
    return tf.nn.relu(x)

  def _fully_connected(self, x, out_dim):
    with tf.name_scope('fully_connected') as name_scope:
      x = self._apply_custom_weight_layer(x, CustomWeightDenseLayer, out_dim)

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _avg_pool(self, x, pool_size, stride, padding='SAME'):
    with tf.name_scope('avg_pool') as name_scope:
      x = tf.layers.average_pooling2d(
          x, pool_size, stride, padding, data_format=self._data_format)

    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _global_avg_pool(self, x):
    with tf.name_scope('global_avg_pool') as name_scope:
      assert x.get_shape().ndims == 4
      if self._data_format == 'channels_first':
        x = tf.reduce_mean(x, [2, 3])
      else:
        x = tf.reduce_mean(x, [1, 2])
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x
