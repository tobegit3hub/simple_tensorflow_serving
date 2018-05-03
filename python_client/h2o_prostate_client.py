#!/usr/bin/env python

import json
import requests


def main():

  endpoint = "http://127.0.0.1:8500"

  # Load json file
  json_filename = "prostate_test_data.json"

  with open(json_filename, "r") as f:
    pandas_json_data = json.load(f)
    json_data = json.loads(pandas_json_data)

  input_data = {
      "model_name": "default",
      "model_version": 1,
      "data": {
          "data": json_data
      }
  }
  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
