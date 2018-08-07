#!/usr/bin/env python

import requests
import base64


def main():

  image_file_name = "../images/mew.jpg"
  image_b64_string = base64.urlsafe_b64encode(
      open(image_file_name, "rb").read())

  endpoint = "http://127.0.0.1:8500"
  input_data = {
      #"model_name": "tensorflow_template_application_model",
      #"model_version": 2,
      #"signature_name": "serving_base64",
      "data": {
          "images": [image_b64_string]
      }
  }
  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
