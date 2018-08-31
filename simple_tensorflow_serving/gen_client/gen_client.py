
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf

from . import gen_bash
from . import gen_golang
from . import gen_javascript
from . import gen_python

logger = logging.getLogger("simple_tensorflow_serving")


def gen_tensorflow_client(tensorflow_inference_service, language, model_name):
  """
  Generate the TensorFlow client for programming languages.
  
  Args:
    tensorflow_inference_service: The tensorflow service object.
    language: The client in this programming language to generate.
    model_name: The name of the model.
    
  Return:
    None
  """

  if language not in ["json", "bash", "python", "golang", "javascript"]:
    logger.error(
        "Language: {} is not supported to gen client".format(language))
    return

  # Example: {"keys": [-1, 1], "features": [-1, 9]}
  input_opname_shape_map = {}
  input_opname_dtype_map = {}

  for input_item in tensorflow_inference_service.model_graph_signature.inputs.items(
  ):
    # Example: "keys"
    input_opname = input_item[0]
    input_opname_shape_map[input_opname] = []
    input_opname_dtype_map[input_opname] = input_item[1].dtype

    # Example: [-1, 1]
    shape_dims = input_item[1].tensor_shape.dim

    for dim in shape_dims:
      input_opname_shape_map[input_opname].append(int(dim.size))

  logger.debug(
      "The input operator and shape: {}".format(input_opname_shape_map))

  # Example: {"keys": [[1.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
  generated_tensor_data = {}

  batch_size = 2
  for opname, shapes in input_opname_shape_map.items():

    # Use to generated the nested array
    internal_array = None

    # Travel all the dims in reverse order
    for i in range(len(shapes)):
      dim = shapes[len(shapes) - 1 - i]

      if dim == -1:
        dim = batch_size

      if internal_array == None:
        # Fill with default values by the types, refer to https://www.tensorflow.org/api_docs/python/tf/DType
        default_value = 1.0
        dtype = input_opname_dtype_map[opname]
        if dtype == int(tf.int8) or dtype == int(tf.uint8) or dtype == int(
            tf.int16) or dtype == int(tf.uint16) or dtype == int(
                tf.int32) or dtype == int(tf.uint32):
          default_value = 1
        elif dtype == int(tf.int64) or dtype == int(tf.uint64):
          default_value = 1
        elif dtype == int(tf.int32):
          default_value = True
        elif dtype == int(tf.string):
          default_value = ""

        internal_array = [default_value for i in range(dim)]

      else:
        internal_array = [internal_array for i in range(dim)]

    generated_tensor_data[opname] = internal_array

  if language == "json":
    return {"data": generated_tensor_data}
  elif language == "bash":
    return gen_bash.gen_tensorflow_client_string(generated_tensor_data, model_name)
  elif language == "python":
    return gen_python.gen_tensorflow_client_string(generated_tensor_data, model_name)
  elif language == "golang":
    return gen_golang.gen_tensorflow_client_string(generated_tensor_data, model_name)
  elif language == "javascript":
    return gen_javascript.gen_tensorflow_client_string(generated_tensor_data, model_name)

  # TODO: Return the string instead of writing in local files
  """
  if language == "bash":
    gen_bash.gen_tensorflow_client(generated_tensor_data, model_name)
  elif language == "python":
    gen_python.gen_tensorflow_client(generated_tensor_data, model_name)
  elif language == "golang":
    gen_golang.gen_tensorflow_client(generated_tensor_data, model_name)
  elif language == "javascript":
    gen_javascript.gen_tensorflow_client(generated_tensor_data, model_name)
  """
