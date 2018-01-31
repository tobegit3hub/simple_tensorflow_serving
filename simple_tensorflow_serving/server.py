#!/usr/bin/env python

import json
import logging

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
flags.DEFINE_boolean("reload_models", True, "Reload models or not")
flags.DEFINE_boolean("verbose", True, "Enable verbose log or not")
flags.DEFINE_string("gen_sdk", "", "Generate the SDK code")
FLAGS = flags.FLAGS

logging.basicConfig(level=logging.DEBUG)
if FLAGS.enable_colored_log:
  import coloredlogs
  coloredlogs.install()
#logging.debug(FLAGS.__flags)


def main():
  # Initialize TensorFlow inference service to load models
  inferenceService = TensorFlowInferenceService(FLAGS.model_base_path,
                                                FLAGS.verbose)

  # Generate sdk code and exit or not
  if FLAGS.gen_sdk != "":
    gen_sdk.gen_tensorflow_sdk(inferenceService, FLAGS.gen_sdk)
    return

  # Start thread to periodically reload models or not
  if FLAGS.reload_models == True:
    inferenceService.dynmaically_reload_models()

  # Initialize flask application
  app = Flask(__name__, template_folder='templates')

  # The API to render the dashboard page
  @app.route("/", methods=["GET"])
  def index():
    return render_template(
        "index.html",
        model_versions=inferenceService.version_session_map.keys(),
        model_graph_signature=str(inferenceService.model_graph_signature))

  # The API to rocess inference request
  @app.route("/", methods=["POST"])
  def inference():
    json_data = json.loads(request.data)
    result = inferenceService.inference(json_data)
    return str(result)

  # Start the HTTP server
  app.run(host=FLAGS.host, port=FLAGS.port)


if __name__ == "__main__":
  main()
