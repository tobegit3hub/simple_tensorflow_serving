from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import tensorflow as tf

logger = logging.getLogger("simple_tensorflow_serving")


def download_hdfs_moels(model_base_path):
  """
  Download the model file from HDFS into local filesystem.
  
  Args:
    model_base_path: The model path which is in local or HDFS.
  
  Return:
    The new path in local of model file.
  """

  if not model_base_path.startswith("hdfs://"):
    return model_base_path
  else:
    model_base_path = update_hdfs_prefix_for_libhdfs(model_base_path)
    new_model_base_path = os.path.join("/tmp/", model_base_path.split("/")[-1])
    logging.info("Copy model file {} to {}".format(model_base_path,
                                                   new_model_base_path))
    tf.gfile.Copy(model_base_path, new_model_base_path, overwrite=True)

    return new_model_base_path


def update_hdfs_prefix_for_libhdfs(model_base_path):
  """
  Update the hdfs path with Java prefix into Libhdfs prefix.
  
  Args:
    model_base_path: The model path which is in local or HDFS.

  Return:
    The new path with Libhdfs prefix.
  """

  if model_base_path.startswith("hdfs:///"):
    # Change "hdfs:///..." to "hdfs://default/..."
    new_model_base_path = model_base_path[:7] + "default" + model_base_path[7:]
    logging.info("Update hdfs prefix from {} to {}".format(
        model_base_path, new_model_base_path))
    return new_model_base_path
  else:
    return model_base_path
