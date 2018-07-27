#!/usr/bin/env python

import requests


def main():

  #endpoint = "http://127.0.0.1:8501/v1/models/default/versions/2:predict"
  #endpoint = "http://127.0.0.1:8501/v1/models/default:predict"
  endpoint = "http://172.27.128.107:8894/v1/models/default:predict"

  input_data = {
      #"signature_name": "serving_default",
      # "keys": [[1.0], [2.0]],
      # "features": [[0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]]
      "instances": [{
          "keys": 1,
          "features": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      }, {
          "keys": 2,
          "features": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      }]
  }

  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
