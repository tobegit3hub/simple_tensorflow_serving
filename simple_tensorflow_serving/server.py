#!/usr/bin/env python

import json
import logging
import pprint

import tensorflow as tf
from flask import Flask, render_template, request

from gen_sdk import gen_sdk
from tensorflow_inference_service import TensorFlowInferenceService

# Define parameters
flags = tf.app.flags
flags.DEFINE_boolean("enable_colored_log", False, "Enable colored log")
flags.DEFINE_string("host", "0.0.0.0", "The host of the server")
flags.DEFINE_integer("port", 8500, "The port of the server")
flags.DEFINE_string("model_base_path", "./model", "The file path of the model")
flags.DEFINE_string("model_name", "default", "The name of the model")
flags.DEFINE_boolean("verbose", True, "Enable verbose log or not")
flags.DEFINE_string("gen_sdk", "", "Generate the SDK code")
FLAGS = flags.FLAGS

logging.basicConfig(level=logging.DEBUG)
if FLAGS.enable_colored_log:
  import coloredlogs
  coloredlogs.install()
pprint.PrettyPrinter().pprint(FLAGS.__flags)


def main():
  # Initialize TensorFlow inference service
  inferenceService = TensorFlowInferenceService(FLAGS.model_base_path,
                                                FLAGS.verbose)

  if FLAGS.gen_sdk != "":
    model_version = inferenceService.get_one_model_version()
    inferenceService.load_saved_model_version(model_version)
    gen_sdk.gen_tensorflow_sdk(inferenceService, FLAGS.gen_sdk)

    return

  inferenceService.dynmaically_reload_models()

  # Initialize flask application
  #app = Flask(__name__)
  app = Flask(__name__, template_folder='templates')

  # Define APIs
  @app.route("/", methods=["GET"])
  def index():
    #return "API Test"
    return render_template('client.py')

  @app.route("/", methods=["POST"])
  def inference():
    json_data = json.loads(request.data)
    result = inferenceService.inference(json_data)
    return str(result)

  # Start HTTP server
  app.run(host=FLAGS.host, port=FLAGS.port)


if __name__ == "__main__":
  main()
