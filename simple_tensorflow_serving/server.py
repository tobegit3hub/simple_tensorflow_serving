#!/usr/bin/env python

import argparse
import argcomplete
import cStringIO
import json
import logging
import sys
from functools import wraps

import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image

from gen_client import gen_client
from tensorflow_inference_service import TensorFlowInferenceService

logging.basicConfig(level=logging.DEBUG)

# Define parameters
parser = argparse.ArgumentParser()

# TODO: Remove if it does not need gunicorn
parser.add_argument(
    "--bind",
    default="0.0.0.0:8500",
    help="Bind address of the server(eg. 0.0.0.0:8500)")
parser.add_argument(
    "--host", default="0.0.0.0", help="The host of the server(eg. 0.0.0.0)")
parser.add_argument(
    "--port", default=8500, help="The port of the server(eg. 8500)", type=int)
parser.add_argument(
    "--model_base_path",
    default="./model",
    help="The file path of the model(eg. 8500)")
parser.add_argument(
    "--model_name",
    default="default",
    help="The name of the model(eg. default)")
parser.add_argument(
    "--reload_models",
    default=True,
    help="Reload models or not(eg. True)",
    type=bool)
parser.add_argument(
    "--custom_op_paths",
    default="",
    help="The path of custom op so files(eg. ./)")
parser.add_argument(
    "--verbose",
    default=True,
    help="Enable verbose log or not(eg. True)",
    type=bool)
parser.add_argument(
    "--gen_client", default="", help="Generate the client code(eg. python)")
parser.add_argument(
    "--enable_auth",
    default=False,
    help="Enable basic auth or not(eg. False)",
    type=bool)
parser.add_argument(
    "--auth_username",
    default="admin",
    help="The username of basic auth(eg. admin)")
parser.add_argument(
    "--auth_password",
    default="admin",
    help="The password of basic auth(eg. admin)")
parser.add_argument(
    "--enable_colored_log",
    default=False,
    help="Enable colored log(eg. False)",
    type=bool)

# For auto-complete
argcomplete.autocomplete(parser)

if len(sys.argv) == 1:
  args = parser.parse_args(["-h"])
  args.func(args)
else:
  args = parser.parse_args(sys.argv[1:])
  #import ipdb;ipdb.set_trace()

  for arg in vars(args):
    logging.info("{}: {}".format(arg, getattr(args, arg)))

  if args.enable_colored_log:
    import coloredlogs
    coloredlogs.install()


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
  if args.enable_auth:
    if username == args.auth_username and password == args.auth_password:
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
  def decorated(*decorator_args, **decorator_kwargs):

    auth = request.authorization

    if args.enable_auth:
      if not auth or not verify_authentication(auth.username, auth.password):
        response = Response("Need basic auth to request the resources", 401, {
            "WWW-Authenticate": '"Basic realm="Login Required"'
        })
        return response

    return f(*decorator_args, **decorator_kwargs)

  return decorated


# Initialize flask application
application = Flask(__name__, template_folder='templates')

# Initialize TensorFlow inference service to load models
inferenceService = TensorFlowInferenceService(
    args.model_base_path, args.custom_op_paths, args.verbose)

# Generate client code and exit or not
if args.gen_client != "":
  gen_client.gen_tensorflow_client(inferenceService, args.gen_client)
  exit(0)

# Start thread to periodically reload models or not
if args.reload_models == True:
  inferenceService.dynmaically_reload_models()


# The API to render the dashboard page
@application.route("/", methods=["GET"])
@requires_auth
def index():
  return render_template(
      "index.html",
      model_versions=inferenceService.version_session_map.keys(),
      model_graph_signature=str(inferenceService.model_graph_signature))


# The API to rocess inference request
@application.route("/", methods=["POST"])
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
    logging.error("Unsupported content type: {}".format(request.content_type))
    return "Error, unsupported content type"

  # Request backend service with json data
  #logging.debug("Constructed request data as json: {}".format(json_data))
  result = inferenceService.inference(json_data)

  # TODO: Change the decoder for numpy data
  return jsonify(json.loads(json.dumps(result, cls=NumpyEncoder)))


def main():
  # Start the HTTP server
  application.run(host=args.host, port=args.port)


if __name__ == "__main__":
  main()
