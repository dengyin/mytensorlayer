#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep learning and Reinforcement learning library for Researchers and Engineers"""

import os
from distutils.version import LooseVersion

from my_tensorlayer.package_info import VERSION
from my_tensorlayer.package_info import __shortversion__
from my_tensorlayer.package_info import __version__

from my_tensorlayer.package_info import __package_name__
from my_tensorlayer.package_info import __contact_names__
from my_tensorlayer.package_info import __contact_emails__
from my_tensorlayer.package_info import __homepage__
from my_tensorlayer.package_info import __repository_url__
from my_tensorlayer.package_info import __download_url__
from my_tensorlayer.package_info import __description__
from my_tensorlayer.package_info import __license__
from my_tensorlayer.package_info import __keywords__

if 'my_tensorlayer_PACKAGE_BUILDING' not in os.environ:

    try:
        import tensorflow
    except Exception as e:
        raise ImportError(
            "Tensorflow is not installed, please install it with the one of the following commands:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

    if ("SPHINXBUILD" not in os.environ and "READTHEDOCS" not in os.environ and
            LooseVersion(tensorflow.__version__) < LooseVersion("1.6.0")):
        raise RuntimeError(
            "my_tensorlayer does not support Tensorflow version older than 1.6.0.\n"
            "Please update Tensorflow with:\n"
            " - `pip install --upgrade tensorflow`\n"
            " - `pip install --upgrade tensorflow-gpu`"
        )

    from my_tensorlayer import activation
    from my_tensorlayer import array_ops
    from my_tensorlayer import cost
    from my_tensorlayer import decorators
    from my_tensorlayer import files
    from my_tensorlayer import initializers
    from my_tensorlayer import iterate
    from my_tensorlayer import layers
    from my_tensorlayer import lazy_imports
    from my_tensorlayer import logging
    from my_tensorlayer import models
    from my_tensorlayer import optimizers
    from my_tensorlayer import rein

    from my_tensorlayer.lazy_imports import LazyImport

    # Lazy Imports
    db = LazyImport("my_tensorlayer.db")
    distributed = LazyImport("my_tensorlayer.distributed")
    nlp = LazyImport("my_tensorlayer.nlp")
    prepro = LazyImport("my_tensorlayer.prepro")
    utils = LazyImport("my_tensorlayer.utils")
    visualize = LazyImport("my_tensorlayer.visualize")

    # alias
    act = activation
    vis = visualize

    alphas = array_ops.alphas
    alphas_like = array_ops.alphas_like

    # global vars
    global_flag = {}
    global_dict = {}
