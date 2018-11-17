#!/usr/bin/env python

import requests


def main():

  endpoint = "http://127.0.0.1:8500"
  input_data = {
      "model_name": "default",
      "model_version": 1,
      "data": {
          "format": "libsvm",
          "max_ids": 692,
          "ids": [128, 129, 130],
          "values": [51, 159, 20]
      }
  }
  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
