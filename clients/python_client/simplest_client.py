#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import requests


def main():

  endpoint = "http://127.0.0.1:8500"

  print("Request for raw model signature")
  input_data = {"data": {"keys": [1, 2]}}
  result = requests.post(endpoint, json=input_data)
  print(result.text)

  print("Request with preprocess")
  input_data = {"preprocess": True, "data": {"keys": ["你好世界", "机器学习预处理模型"]}}
  result = requests.post(endpoint, json=input_data)
  print(result.text)

  print("Request with preprocess and postprocess")
  input_data = {
      "preprocess": True,
      "postprocess": True,
      "data": {
          "keys": ["你好世界", "机器学习预处理模型"]
      }
  }
  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
