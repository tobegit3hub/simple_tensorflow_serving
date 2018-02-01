#!/usr/bin/env python

import requests


def main():
  endpoint = "http://127.0.0.1:8500"

  files = {"image": open("../examples/example.jpg", "rb")}
  data = {"model_version": 1}
  response = requests.post(endpoint, files=files, data=data)
  print(response.text)


if __name__ == "__main__":
  main()
