#!/usr/bin/env python

import requests


def main():

  endpoint = 'http://localhost:8080/openscoring/model/PmmlModel'

  input_data = {"id": None, "arguments": {"Sepal_Width": 3.5, "Petal_Width": 0.2, "Sepal_Length": 5.1, "Petal_Length": 1.4}}

  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
