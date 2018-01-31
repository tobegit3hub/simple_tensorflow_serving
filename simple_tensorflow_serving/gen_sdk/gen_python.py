import json
import logging

from jinja2 import Template


def gen_tensorflow_python_sdk(input_opname_shape_map):
  """
  Generate TensorFlow SDK in Python.

  Args:
    input_opname_shape_map: Example is {"keys": [-1, 1], "features": [-1, 9]}
  """

  # TODO: Write into local file

  code_template = """#!/usr/bin/env python

import requests

def main():
  endpoint = "http://127.0.0.1:8500"
  json_data = {"data": {{ tensor_data }} }
  result = requests.post(endpoint, json=json_data)
  print(result.text)

if __name__ == "__main__":
  main()
  """

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
        internal_array = [1.0 for i in range(dim)]
      else:
        internal_array = [internal_array for i in range(dim)]

    generated_tensor_data[opname] = internal_array

  # Example: '{"keys": [[111.0], [112.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}'
  generated_tensor_data_string = json.dumps(generated_tensor_data)
  template = Template(code_template)
  generate_code = template.render(tensor_data=generated_tensor_data_string)

  logging.debug("Generate the code:\n{}".format(generate_code))

  generated_code_filename = "client.py"
  with open(generated_code_filename, "w") as f:
    f.write(generate_code)

  logging.info('Save the generated code in {}, try "python {}"'.format(
      generated_code_filename, generated_code_filename))
