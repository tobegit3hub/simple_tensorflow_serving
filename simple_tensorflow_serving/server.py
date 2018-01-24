#!/usr/bin/env python

import json
import logging
import pprint

import tensorflow as tf
from flask import Flask, request

from tensorflow_inference_service import TensorFlowInferenceService

# Define parameters
flags = tf.app.flags
flags.DEFINE_boolean("enable_colored_log", False, "Enable colored log")
flags.DEFINE_string("host", "0.0.0.0", "The host of the server")
flags.DEFINE_integer("port", 8500, "The port of the server")
flags.DEFINE_string("model_base_path", "./model", "The file path of the model")
flags.DEFINE_string("model_name", "default", "The name of the model")
flags.DEFINE_integer("model_version", 1, "The version of the model")
flags.DEFINE_boolean("verbose", True, "Enable verbose log or not")
FLAGS = flags.FLAGS

logging.basicConfig(level=logging.DEBUG)
if FLAGS.enable_colored_log:
  import coloredlogs
  coloredlogs.install()
pprint.PrettyPrinter().pprint(FLAGS.__flags)


def main():
  # Initialize TensorFlow inference service
  inferenceService = TensorFlowInferenceService(
      FLAGS.model_base_path, FLAGS.model_name, FLAGS.model_version,
      FLAGS.verbose)

  # Initialize flask application
  app = Flask(__name__)

  @app.route("/", methods=["GET"])
  def index():
    return "Get is not supported"

  @app.route("/", methods=["POST"])
  def inference():
    input_data = json.loads(request.data)
    result = inferenceService.inference(input_data)
    return str(result)

  logging.info(
      "Start the server in host: {}, port: {}".format(FLAGS.host, FLAGS.port))
  app.run(host=FLAGS.host, port=FLAGS.port)


if __name__ == "__main__":
  main()
