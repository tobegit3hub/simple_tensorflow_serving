
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging

from jinja2 import Template


def gen_tensorflow_client_string(generated_tensor_data, model_name):
  """
  Generate TensorFlow SDK in Bash.

  Args:
    generated_tensor_data: Example is {"keys": [[1.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
  """

  code_template = """#!/bin/bash

curl -H "Content-Type: application/json" -X POST -d '{"model_name": "{{ model_name }}", "data": {{ tensor_data }} }' http://127.0.0.1:8500
  """

  generated_tensor_data_string = json.dumps(generated_tensor_data)
  template = Template(code_template)
  generate_code = template.render(
      model_name=model_name, tensor_data=generated_tensor_data_string)
  logging.debug("Generate the code in Bash:\n{}".format(generate_code))

  return generate_code


def gen_tensorflow_client(generated_tensor_data, model_name):

  generate_code = gen_tensorflow_client_string(generated_tensor_data, model_name)

  generated_code_filename = "client.sh"
  with open(generated_code_filename, "w") as f:
    f.write(generate_code)

  logging.info('Save the generated code in {}, try "bash {}"'.format(
      generated_code_filename, generated_code_filename))
