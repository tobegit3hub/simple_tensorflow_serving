#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import requests
import logging
from PIL import Image
import numpy as np


logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

def predict_image(image_file_path, channel_layout="RGB", run_profile="", port=8500):
  endpoint = "http://127.0.0.1:" + str(port)

  img = Image.open(image_file_path)
  img = img.convert(channel_layout)
  img.load()
  image_ndarray = np.asarray(img, dtype="int32")
  # Shape is [48, 400, 3] -> [400, 48, 3]
  image_ndarray = image_ndarray.transpose((1, 0, 2))
  image_array = [image_ndarray.tolist()]
  # TODO: Support specified model name
  json_data = {"model_name": "default",
               "data": {"image": image_array},
               "run_profile": run_profile}

  result = requests.post(endpoint, json=json_data)

  # Shape is [-1, -1]
  predict_result = json.loads(result.text)
  print("Get predict result:{}".format(predict_result))

  return predict_result


def predict_json(json_data, port=8500):
  # TODO: Support for other endpoint
  endpoint = "http://127.0.0.1:" + str(port)
  predict_result = "Error"

  try:
    result = requests.post(endpoint, json=json_data)
    predict_result = result.json()
    logging.debug("Get predict result:{}".format(predict_result))
  except Exception as e:
    logging.error("Get result: {} and exception: {}".format(result, e.message))

  return predict_result


def gen_json_and_clients(model_name="default", signature_name="serving_default", port=8500):
  endpoint = "http://127.0.0.1:{}/v1/models/{}/gen_json".format(port, model_name)
  return_result = "Error"

  try:
    result = requests.get(endpoint)
    return_result = result.json()
    logging.debug("Get predict result:{}".format(return_result))
  except Exception as e:
    logging.error("Get exception: {}".format(e.message))

  return return_result



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
