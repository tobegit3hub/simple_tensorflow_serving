#!/usr/bin/env python

import requests
import time
import base64


def main():
  benchmark("simple_tensorflow_serving_flask")

  time.sleep(3)
  benchmark("simple_tensorflow_serving_uwsgi")

  time.sleep(3)
  benchmark("tensorflow_serving_restful")


def benchmark(benchmark_type):
  print("Start benchmark for {}".format(benchmark_type))

  batch_size = 16
  image_b64_string = base64.urlsafe_b64encode(open("./0.jpg", "rb").read())
  image_b64_strings = [image_b64_string for i in range(batch_size)]

  if benchmark_type == "simple_tensorflow_serving_flask":
    endpoint = "http://127.0.0.1:8500"
    input_data = {
      "data": {
        "images": image_b64_strings
      }
    }

  elif benchmark_type == "simple_tensorflow_serving_uwsgi":
    endpoint = "http://127.0.0.1:8501"
    input_data = {
      "data": {
        "images": image_b64_strings
      }
    }

  elif benchmark_type == "tensorflow_serving_restful":
    endpoint = "http://127.0.0.1:8503/v1/models/default/versions/1:predict"
    image_b64_strings = [{"images": image_b64_string} for i in range(batch_size)]
    input_data = {
      "instances": image_b64_strings
    }

  start_time = time.time()
  for i in range(10):
    result = requests.post(endpoint, json=input_data)
  end_time = time.time()
  print("Cost time: {}".format(end_time - start_time))
  print(result)
  #print(result.text)


if __name__ == "__main__":
  main()
