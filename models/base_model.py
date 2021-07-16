from my_tensorlayer import layers
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod


class BaseModel(metaclass=ABCMeta):
    @abstractmethod
    def build_graph(self, name, x, is_train):
        pass

    @property
    def trn_graph(self):
        assert hasattr(self, 'input')
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            return self.build_graph(self.name, self.input, is_train=True)

    @property
    def predict_graph(self):
        assert hasattr(self, 'input')
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            return self.build_graph(self.name, self.input, is_train=False)

    @property
    def params(self):
        return layers.get_variables_with_name(self.name, True, False)

    @abstractmethod
    def create_feeddict(self, x):
        result = dict()
        return result

    def __init__(self, name, output_dim):
        self.name = name
        self.output = tf.placeholder(tf.int32, shape=[None, output_dim])
        self.sess = tf.InteractiveSession()
        self.train_epoch = 0

    def predict_raw(self, x, batch_size=32):
        x = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        iterator = x.make_initializable_iterator()
        batch_x = iterator.get_next()
        result = None
        self.sess.run(iterator.initializer)
        while True:
            try:
                batch_train_x_input = self.sess.run(batch_x)
                batch_train_y_pred = self.sess.run(self.predict_graph.outputs,
                                                   feed_dict=self.create_feeddict(batch_train_x_input))
                if result:
                    np.vstack((result, batch_train_y_pred))
                else:
                    result = batch_train_y_pred
            except tf.errors.OutOfRangeError:
                break
        return result

    def fit(self, train_x, train_y, batch_size=32, n_epoch=1, optimizer=tf.train.AdamOptimizer(), print_freq=1,
            val_dataset=None, val_name=None):
        train_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.output,
                                                            logits=self.trn_graph.outputs)
        train_op = optimizer.minimize(train_loss, var_list=self.params)

        train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_data = train_data.shuffle(1000).batch(batch_size)
        iterator = train_data.make_initializable_iterator()
        batch_train = iterator.get_next()

        if self.train_epoch == 0:
            layers.initialize_global_variables(self.sess)
        if not hasattr(self, 'optimizer') or type(self.optimizer) != type(optimizer):
            self.optimizer = optimizer
            self.sess.run(tf.variables_initializer(self.optimizer.variables()))

        for epoch in range(n_epoch):
            self.train_epoch += 1
            self.sess.run(iterator.initializer)
            step = 0
            epoch_train_loss = 0
            epoch_train_count = 0
            while True:
                try:
                    step += 1
                    batch_train_x, batch_train_y = self.sess.run(batch_train)
                    feed_dict = self.create_feeddict(batch_train_x)
                    feed_dict[self.output] = np.reshape(batch_train_y, (-1, 1))
                    _, loss = self.sess.run([train_op, train_loss], feed_dict=feed_dict)
                    print('\r', f'epoch:{self.train_epoch} step:{step}, loss:{loss}', end='\r', flush=True)
                    step_count = feed_dict[self.output].shape[0]
                    epoch_train_loss += loss * step_count
                    epoch_train_count += step_count
                except tf.errors.OutOfRangeError:
                    break
            print(f'epoch:{self.train_epoch}, loss:{epoch_train_loss / epoch_train_count}')
