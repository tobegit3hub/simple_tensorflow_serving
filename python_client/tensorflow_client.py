#!/usr/bin/env python

import requests

def main():
  simple_tensorflow_serving_endpoint = "http://127.0.0.1:8500"

  payload = {"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
  result = requests.post(simple_tensorflow_serving_endpoint, json=payload)
  print(result.text)

if __name__ == "__main__":
  main()
