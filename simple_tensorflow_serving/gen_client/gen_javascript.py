import json
import logging

from jinja2 import Template


def gen_tensorflow_client(generated_tensor_data, model_name):
  """
  Generate TensorFlow SDK in JavaScript.

  Args:
    generated_tensor_data: Example is {"keys": [[1.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
  """

  code_template = """#!/usr/bin/env node

var request = require("request");

var options = {
    uri: "http://127.0.0.1:8500",
    method: "POST",
    json: {"model_name": "{{ model_name }}", "data": {{ tensor_data }} }
};

request(options, function (error, response, body) {
    if (!error && response.statusCode == 200) {
        console.log(body)
    } else {
        console.log(error)
    }
});
  """

  generated_tensor_data_string = json.dumps(generated_tensor_data)
  template = Template(code_template)
  generate_code = template.render(model_name=model_name, tensor_data=generated_tensor_data_string)
  logging.debug("Generate the code in JavaScript:\n{}".format(generate_code))

  generated_code_filename = "client.js"
  with open(generated_code_filename, "w") as f:
    f.write(generate_code)

  logging.info(
      'Save the generated code in {}, try "npm install request && node {}"'.
      format(generated_code_filename, generated_code_filename))
