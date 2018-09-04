from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import marshal
import types
import logging

logger = logging.getLogger("simple_tensorflow_serving")


def get_functrion_from_marshal_file(function_file_path,
                                    function_name="function"):
  logging.info("Try to get function from file: {}".format(function_file_path))

  function_object = None

  if os.path.exists(function_file_path):
    with open(function_file_path, "r") as f:
      preprocess_function_string = f.read()
      loaded_function = marshal.loads(preprocess_function_string)
      function_object = types.FunctionType(loaded_function,
                                           globals(), function_name)

  return function_object


def get_preprocess_postprocess_function_from_model_path(model_base_path):
  # Example: ./models/scikitlearn_iris/model.joblib
  model_parent_path = "/".join(model_base_path.split("/")[:-1])

  preprocess_file_path = os.path.join(model_parent_path,
                                      "preprocess_function.marshal")
  preprocess_function = get_functrion_from_marshal_file(preprocess_file_path)

  postprocess_file_path = os.path.join(model_parent_path,
                                       "postprocess_function.marshal")
  postprocess_function = get_functrion_from_marshal_file(postprocess_file_path)

  return preprocess_function, postprocess_function
