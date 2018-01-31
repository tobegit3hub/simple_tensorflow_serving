import logging

import gen_python


def gen_tensorflow_sdk(tensorflow_inference_service, language):
  """
  Generate the TensorFlow SDK for programming languages.
  
  Args:
    tensorflow_inference_service: The tensorflow service object.
    language: The sdk in this programming language to generate.
    
  Return:
    None
  """

  if language not in ["python"]:
    logging.error("Language: {} is not supported to gen sdk".format(language))
    return

  if language == "python":

    # Example: {'keys': [-1, 1], 'features': [-1, 9]}
    input_opname_shape_map = {}

    for input_item in tensorflow_inference_service.model_graph_signature.inputs.items(
    ):
      # Example: "keys"
      input_opname = input_item[0]
      input_opname_shape_map[input_opname] = []

      # Example: [-1, 1]
      shape_dims = input_item[1].tensor_shape.dim

      for dim in shape_dims:
        input_opname_shape_map[input_opname].append(int(dim.size))

    logging.debug(
        "The input operator and shape: {}".format(input_opname_shape_map))

    gen_python.gen_tensorflow_python_sdk(input_opname_shape_map)
