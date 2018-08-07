#!/usr/bin/env python

import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat

model_path = "model"
model_version = 1

keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1], name="keys")
keys_identity = tf.identity(keys_placeholder, name="inference_keys")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

model_signature = signature_def_utils.build_signature_def(
    inputs={
        "keys": utils.build_tensor_info(keys_placeholder),
    },
    outputs={
        "keys": utils.build_tensor_info(keys_identity),
    },
    method_name=signature_constants.PREDICT_METHOD_NAME)

export_path = os.path.join(
    compat.as_bytes(model_path), compat.as_bytes(str(model_version)))

builder = saved_model_builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
    sess, [tag_constants.SERVING],
    clear_devices=True,
    signature_def_map={
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: model_signature,
    })

builder.save()
