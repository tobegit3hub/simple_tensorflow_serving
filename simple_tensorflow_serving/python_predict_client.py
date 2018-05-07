#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import requests
from PIL import Image
import numpy as np


def predict_image(image_file_path):
  endpoint = "http://127.0.0.1:8500"

  img = Image.open(image_file_path)
  img = img.convert('RGB')
  img.load()
  image_ndarray = np.asarray(img, dtype="int32")
  # Shape is [48, 400, 3] -> [400, 48, 3]
  image_ndarray = image_ndarray.transpose((1, 0, 2))
  image_array = [image_ndarray.tolist()]
  # TODO: Support specified model name
  json_data = {"model_name": "default", "data": {"image": image_array}}

  result = requests.post(endpoint, json=json_data)

  # Shape is [-1, -1]
  predict_result = json.loads(result.text)
  print("Get predict result:{}".format(predict_result))

  return predict_result


def predict_json(json_data):
  # TODO: Support for other endpoint
  endpoint = "http://127.0.0.1:8500"

  result = requests.post(endpoint, json=json_data)

  predict_result = json.loads(result.text)
  print("Get predict result:{}".format(predict_result))

  return predict_result


# TODO: Only support testing with images
def parse_args():
  parser = argparse.ArgumentParser(description='Predict image')
  parser.add_argument(
      '--image', required=False, type=str, default="./0.jpg", help='The image')
  args = parser.parse_args()
  return args


def main(args):
  image_file_path = args.image
  predict_image(image_file_path)


if __name__ == "__main__":
  main(parse_args())
