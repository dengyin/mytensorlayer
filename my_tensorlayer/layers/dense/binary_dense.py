#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

from my_tensorlayer.layers.core import Layer
from my_tensorlayer.layers.core import LayersConfig

from my_tensorlayer.layers.utils import quantize

from my_tensorlayer import logging

from my_tensorlayer.decorators import deprecated_alias

__all__ = [
    'BinaryDenseLayer',
]


class BinaryDenseLayer(Layer):
    """The :class:`BinaryDenseLayer` class is a binary fully connected layer, which weights are either -1 or 1 while inferencing.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer, usually set to ``tf.act.sign`` or apply :class:`SignLayer` after :class:`BatchNormLayer`.
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inference. (TODO).
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : a str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_units=100,
            act=None,
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='binary_dense',
    ):
        super(BinaryDenseLayer, self
             ).__init__(prev_layer=prev_layer, act=act, W_init_args=W_init_args, b_init_args=b_init_args, name=name)
        logging.info(
            "BinaryDenseLayer  %s: %d %s" %
            (self.name, n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **self.W_init_args
            )
            # W = tl.act.sign(W)    # dont update ...
            W = quantize(W)
            # W = tf.Variable(W)
            # print(W)

            self.outputs = tf.matmul(self.inputs, W)
            # self.outputs = xnor_gemm(self.inputs, W) # TODO

            if b_init is not None:
                try:
                    b = tf.get_variable(
                        name='b', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args
                    )

                except Exception:  # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, dtype=LayersConfig.tf_dtype, **self.b_init_args)

                self.outputs = tf.nn.bias_add(self.outputs, b, name='bias_add')

            self.outputs = self._apply_activation(self.outputs)

        self._add_layers(self.outputs)

        if b_init is not None:
            self._add_params([W, b])
        else:
            self._add_params(W)
