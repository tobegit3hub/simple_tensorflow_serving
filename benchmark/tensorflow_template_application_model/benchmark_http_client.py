#!/usr/bin/env python

import requests
import time


def main():
  benchmark("simple_tensorflow_serving_flask")

  time.sleep(3)
  benchmark("simple_tensorflow_serving_uwsgi")

  time.sleep(3)
  benchmark("tensorflow_serving_restful")


def benchmark(benchmark_type):
  print("Start benchmark for {}".format(benchmark_type))

  if benchmark_type == "simple_tensorflow_serving_flask":
    endpoint = "http://127.0.0.1:8500"
    input_data = {"data": {"keys": [[1]], "features": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}}

  elif benchmark_type == "simple_tensorflow_serving_uwsgi":
    endpoint = "http://127.0.0.1:8501"
    input_data = {"data": {"keys": [[1]], "features": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}}

  elif benchmark_type == "tensorflow_serving_restful":
    endpoint = "http://127.0.0.1:8503/v1/models/default/versions/1:predict"
    input_data = {"instances": [{"keys": 1, "features": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}]}

  start_time = time.time()
  for i in range(100):
    result = requests.post(endpoint, json=input_data)
  end_time = time.time()
  print("Cost time: {}".format(end_time - start_time))
  print(result)


if __name__ == "__main__":
  main()
