import logging


def gen_tensorflow_python_sdk():
  """
  Generate TensorFlow SDK in Python.
  """

  # TODO: Support the Jinjia template
  # TODO: Write into local file
  # TODO: Support reading TensorFlow graph

  code = """
#!/usr/bin/env python

import requests

def main():
    endpoint = "http://127.0.0.1:8500"

    input_data = {"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
    result = requests.post(endpoint, json=input_data)
    print(result.text)

if __name__ == "__main__":
    main()
  """
  print(code)
