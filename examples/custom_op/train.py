#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat


def main():
  # Load custom op
  filename = os.path.join(os.path.dirname(__file__), "zero_out.so")
  zero_out_module = tf.load_op_library(filename)
  zero_out = zero_out_module.zero_out

  # Prepare train data
  train_data = np.ones((2, 2))
  print("Input data: {}".format(train_data))

  # Define the model
  input = tf.placeholder(tf.int32, shape=(None, 2))
  output = zero_out(input)

  # Export the model
  model_path = "model"
  model_version = 1
  model_signature = signature_def_utils.build_signature_def(
      inputs={
          "input": utils.build_tensor_info(input),
      },
      outputs={
          "output": utils.build_tensor_info(output),
      },
      method_name=signature_constants.PREDICT_METHOD_NAME)
  export_path = os.path.join(
      compat.as_bytes(model_path), compat.as_bytes(str(model_version)))
  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

  # Create session to run
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    output_data = sess.run(output, feed_dict={input: train_data})
    print("Output data: {}".format(output_data))

    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            model_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()


if __name__ == "__main__":
  main()
