#!/usr/bin/env python

import requests


def main():

  endpoint = "http://127.0.0.1:8500"
  input_data = {
      "model_name": "default",
      "model_version": 1,
      "data": {
          "data": [[12.0, 2.0]]
      }
  }
  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
