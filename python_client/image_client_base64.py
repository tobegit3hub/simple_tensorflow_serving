#!/usr/bin/env python

import requests
import base64


def base64urlencode(arg):
  stripped = arg.split("=")[0]
  filtered = stripped.replace("+", "-").replace("/", "_")
  return filtered


def base64urldecode(arg):
  filtered = arg.replace("-", "+").replace("_", "/")
  padded = filtered + "=" * ((len(filtered) * -1) % 4)
  return padded


def main():

  #image_b64_string =  open("/Users/tobe/Desktop/b.jpg", "rb").read()
  image_b64_string = base64.b64encode(
      open("/Users/tobe/Desktop/dog.jpeg", "rb").read())
  image_b64_string = base64urlencode(image_b64_string)

  endpoint = "http://127.0.0.1:8500"
  input_data = {
      #"model_name": "tensorflow_template_application_model",
      #"model_version": 2,
      "signature_name": "serving_base64",
      "data": {
          "image_base64": image_b64_string
      }
  }
  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
