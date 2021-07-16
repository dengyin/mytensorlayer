from my_tensorlayer import layers
from .base_model import BaseModel
import tensorflow as tf
import numpy as np


class DnnModel(BaseModel):
    def __init__(self, name, features, output_dim=1):
        super(DnnModel, self).__init__(name, output_dim=output_dim)
        self.input = {
            feature_name: tf.placeholder(tf.float32, shape=[None, 1], name=feature_name) for feature_name in features
        }

    def build_graph(self, name, x, is_train):
        network = [
            layers.ReshapeLayer(layers.InputLayer(x[k], name=f'input_{k}'), shape=(-1, 1),
                                name=f'reshape_{k}')
            for k in x.keys()]
        network = layers.ConcatLayer(network, concat_dim=1)
        network = layers.DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop1')
        network = layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu1')
        network = layers.DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop2')
        network = layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu2')
        network = layers.DropoutLayer(network, keep=0.5, is_fix=True, is_train=is_train, name='drop3')
        network = layers.DenseLayer(network, n_units=3, act=None, name='output')
        return network

    def create_feeddict(self, x):
        result = dict()
        for featurn_name in self.input.keys():
            result[self.input.get(featurn_name)] = np.reshape(x[featurn_name], (-1, 1))
        return result