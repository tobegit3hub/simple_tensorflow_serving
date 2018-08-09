#!/usr/bin/env python

import requests
import time


def main():

  endpoint = "http://127.0.0.1:8500"
  aendpoint = "http://127.0.0.1:8501"
  aendpoint = "http://127.0.0.1:8503"
  input_data = {
      "data": {
          "keys": [[1]],
          "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1]]
      }
  }


  ainput_data = {
      "instances": {
          "keys": 1,
          "features": [1, 1, 1, 1, 1, 1, 1, 1, 1]
      }
  }


  start_time = time.time()
  for i in range(10000):
    result = requests.post(endpoint, json=input_data)
  print("Cost time: {}".format(time.time() - start_time))


  print(result.text)


if __name__ == "__main__":
  main()
