#!/usr/bin/env python

import requests
import base64


def base64_to_url_safe_base64(arg):
  stripped = arg.split("=")[0]
  filtered = stripped.replace("+", "-").replace("/", "_")
  return filtered


def url_safe_base64_to_base64(arg):
  filtered = arg.replace("-", "+").replace("_", "/")
  padded = filtered + "=" * ((len(filtered) * -1) % 4)
  return padded


def main():

  image_b64_string = base64.urlsafe_b64encode(open("/Users/tobe/Desktop/dog.jpeg", "rb").read())
  image_b64_string2 = base64.urlsafe_b64encode(open("/Users/tobe/Desktop/cat.jpeg", "rb").read())
  image_b64_string3 = base64.urlsafe_b64encode(open("/Users/tobe/Desktop/car.jpeg", "rb").read())
  #image_b64_strings = [image_b64_string, image_b64_string2, image_b64_string3]
  image_b64_strings = [image_b64_string, image_b64_string2, image_b64_string3]

  endpoint = "http://127.0.0.1:8500"
  input_data = {
      #"model_name": "tensorflow_template_application_model",
      #"model_version": 2,
      #"signature_name": "serving_base64",
      "data": {
          "images": image_b64_strings
      }
  }
  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
