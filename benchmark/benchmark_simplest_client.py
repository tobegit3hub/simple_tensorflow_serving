#!/usr/bin/env python

import time
import requests

def main():
  endpoint = "http://127.0.0.1:8500"
  json_data = {"model_name": "default", "data": {"keys": [1, 1]} }

  iteration = 1000
  start_time = time.time()

  for i in range(iteration):
    result = requests.post(endpoint, json=json_data)

  end_time = time.time()
  print(result.json())

  print("Benchmark iteration: {}, time: {}".format(iteration, end_time - start_time))


if __name__ == "__main__":
  main()
  
