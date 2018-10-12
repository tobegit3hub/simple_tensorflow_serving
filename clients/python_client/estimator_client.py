#!/usr/bin/env python

import requests
import base64


def main():

  string_data1 = '\n\x1f\n\x0e\n\x01a\x12\t\n\x07\n\x05hello\n\r\n\x01b\x12\x08\x12\x06\n\x04\x00\x00\x00?'
  string_data2 = '\n \n\x0f\n\x01a\x12\n\n\x08\n\x06\xe4\xbd\xa0\xe5\xa5\xbd\n\r\n\x01b\x12\x08\x12\x06\n\x04\x00\x00\x80\xbf'
  # Example: Ch8KDgoBYRIJCgcKBWhlbGxvCg0KAWISCBIGCgQAAAA_
  string_data1_b64 = base64.urlsafe_b64encode(string_data1)
  string_data2_b64 = base64.urlsafe_b64encode(string_data2)
  string_datas = [{"b64": string_data1_b64}, {"b64": string_data2_b64}]

  endpoint = "http://127.0.0.1:8500"
  input_data = {"data": {"inputs": string_datas}}
  result = requests.post(endpoint, json=input_data)
  print(result.text)


if __name__ == "__main__":
  main()
