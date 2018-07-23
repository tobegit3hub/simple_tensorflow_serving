#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base64
import tensorflow as tf


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main():
  input_file_name = "data.txt"
  seperator_symbol = " "

  # Example: {'age': _float_feature(value=25), 'workclass': _bytes_feature(value='Private'.encode())}
  feature_dict = {}
  serialized_strings = []

  with open(input_file_name, "r") as f:
    lines = f.readlines()

    keys = [item.strip() for item in lines[0].split(seperator_symbol)]
    types = [item.strip() for item in lines[1].split(seperator_symbol)]

    for i in range(2, len(lines)):
      items = [item.strip() for item in lines[i].split(seperator_symbol)]

      for j in range(len(items)):
        item = items[j]
        if types[j] == "float":
          item = float(item)
          feature_dict[keys[j]] = _float_feature(value=item)
        elif types[j] == "string":
          feature_dict[keys[j]] = _bytes_feature(value=item)

      example = tf.train.Example(features=tf.train.Features(
          feature=feature_dict))
      serialized = example.SerializeToString()
      serialized_strings.append(serialized)

    serialized_proto = tf.contrib.util.make_tensor_proto(
        serialized_strings, dtype=tf.string)
    serialized_proto_handle = serialized_proto.string_val

    # Example: "\n\x1f\n\x0e\n\x01a\x12\t\n\x07\n\x05hello\n\r\n\x01b\x12\x08\x12\x06\n\x04\x00\x00\x00?"
    proto_string = serialized_proto_handle.pop()
    base64_proto_string = base64.urlsafe_b64encode(proto_string)
    print("Base64 string: {}".format(base64_proto_string))


if __name__ == "__main__":
  main()
