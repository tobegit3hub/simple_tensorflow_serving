#!/usr/bin/env python

import requests


def main():

  endpoint = "http://127.0.0.1:8500"
  input_data = {
      "model_name": "default",
      "model_version": 1,
      "data": {
          "Sepal_Length": 5.1,
          "Sepal_Width": 3.5,
          "Petal_Length": 1.4,
          "Petal_Width": 0.2
      }
  }
  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
