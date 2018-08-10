#!/usr/bin/env python

import requests
import time


def main():
  benchmark(1)

  time.sleep(3)
  benchmark(16)

  time.sleep(3)
  benchmark(128)

  time.sleep(3)
  benchmark(1024)

  time.sleep(3)
  benchmark(8192)

  time.sleep(3)
  benchmark(65536)

  time.sleep(3)
  benchmark(524288)

  time.sleep(3)
  benchmark(4194304)


def benchmark(batch_size):
  print("Start benchmark for batch size: {}".format(batch_size))

  endpoint = "http://127.0.0.1:8500"

  batch_data = [1 for i in range(batch_size)]
  input_data = {"data": {"keys": [batch_data]}}

  if batch_size == 4194304:
    start_time = time.time()
    for i in range(10):
      result = requests.post(endpoint, json=input_data)
    end_time = time.time()
  elif batch_size == 524288 or batch_size == 65536:
    start_time = time.time()
    for i in range(100):
      result = requests.post(endpoint, json=input_data)
    end_time = time.time()
  else:
    start_time = time.time()
    for i in range(1000):
      result = requests.post(endpoint, json=input_data)
    end_time = time.time()

  print("Cost time: {}".format(end_time - start_time))
  print(result)


if __name__ == "__main__":
  main()
