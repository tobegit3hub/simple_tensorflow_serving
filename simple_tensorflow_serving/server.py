#!/usr/bin/env python

import cStringIO
import json
import logging
from functools import wraps

import numpy as np
import tensorflow as tf
from flask import Flask, Response, render_template, request, jsonify
from PIL import Image

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
flags.DEFINE_boolean("enable_auth", False, "Enable basic auth or not")
flags.DEFINE_string("auth_username", "admin", "The username of basic auth")
flags.DEFINE_string("auth_password", "admin", "The password of basic auth")
FLAGS = flags.FLAGS

logging.basicConfig(level=logging.DEBUG)
if FLAGS.enable_colored_log:
  import coloredlogs
  coloredlogs.install()
#logging.debug(FLAGS.__flags)


class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)


def verify_authentication(username, password):
  """
  Verify if this user should be authenticated or not.

  Args:
    username: The user name.
    password: The password.

  Return:
    True if it passes and False if it does not pass.
  """
  if FLAGS.enable_auth:
    if username == FLAGS.auth_username and password == FLAGS.auth_password:
      return True
    else:
      return False
  else:
    return True


def requires_auth(f):
  """
  The decorator to enable basic auth.
  """

  @wraps(f)
  def decorated(*args, **kwargs):

    auth = request.authorization

    if FLAGS.enable_auth:
      if not auth or not verify_authentication(auth.username, auth.password):
        response = Response("Need basic auth to request the resources", 401, {
            "WWW-Authenticate": '"Basic realm="Login Required"'
        })
        return response

    return f(*args, **kwargs)

  return decorated


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
  @requires_auth
  def index():
    return render_template(
        "index.html",
        model_versions=inferenceService.version_session_map.keys(),
        model_graph_signature=str(inferenceService.model_graph_signature))

  # The API to rocess inference request
  @app.route("/", methods=["POST"])
  @requires_auth
  def inference():

    # Process requests with json data
    if request.content_type.startswith("application/json"):
      json_data = json.loads(request.data)

    # Process requests with raw image
    elif request.content_type.startswith("multipart/form-data"):
      json_data = {}

      if "model_version" in request.form:
        json_data["model_version"] = int(request.form["model_version"])

      image_content = request.files["image"].read()
      image_string = np.fromstring(image_content, np.uint8)
      image_string_io = cStringIO.StringIO(image_string)
      image_file = Image.open(image_string_io)
      image_array = np.array(image_file)
      # TODO: Support multiple images without reshaping
      image_array = image_array.reshape(1, *image_array.shape)

      json_data["data"] = {"image": image_array}

    else:
      logging.error(
          "Unsupported content type: {}".format(request.content_type))
      return "Error, unsupported content type"

    # Request backend service with json data
    #logging.debug("Constructed request data as json: {}".format(json_data))
    result = inferenceService.inference(json_data)

    # TODO: Change the decoder for numpy data
    return jsonify(json.loads(json.dumps(result, cls=NumpyEncoder)))


  # Start the HTTP server
  app.run(host=FLAGS.host, port=FLAGS.port)


if __name__ == "__main__":
  main()
