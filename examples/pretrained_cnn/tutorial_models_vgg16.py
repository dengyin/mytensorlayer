#! /usr/bin/python
# -*- coding: utf-8 -*-
"""VGG-16 for ImageNet using TL models."""

import time
import numpy as np
import tensorflow as tf
import my_tensorlayer as tl
from my_tensorlayer.models.imagenet_classes import class_names

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])

# get the whole model
vgg = tl.models.VGG16(x)

# restore pre-trained VGG parameters
sess = tf.InteractiveSession()

vgg.restore_params(sess)

probs = tf.nn.softmax(vgg.outputs)

vgg.print_params(False)

vgg.print_layers()

img1 = tl.vis.read_image('data/tiger.jpeg')
img1 = tl.prepro.imresize(img1, (224, 224))
# rescale pixels values in the range of 0-1
img1 = img1 / 255.0
if ((0 <= img1).all() and (img1 <= 1.0).all()) is False:
    raise Exception("image value should be [0, 1]")

_ = sess.run(probs, feed_dict={x: [img1]})[0]  # 1st time takes time to compile
start_time = time.time()
prob = sess.run(probs, feed_dict={x: [img1]})[0]
print("  End time : %.5ss" % (time.time() - start_time))
preds = (np.argsort(prob)[::-1])[0:5]
for p in preds:
    print(class_names[p], prob[p])
