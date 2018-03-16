#!/usr/bin/env python

import argparse
#import argcomplete
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
from mxnet_inference_service import MxnetInferenceService

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
    "--model_name",
    default="default",
    help="The name of the model(eg. default)")
parser.add_argument(
    "--model_base_path",
    default="./model",
    help="The file path of the model(eg. 8500)")
parser.add_argument(
    "--model_platform",
    default="tensorflow",
    help="The platform of model(eg. tensorflow)")
parser.add_argument(
    "--model_config_file",
    default="",
    help="The file of the model config(eg. '')")
# TODO: type=bool not works, make it true by default if fixing exit bug
parser.add_argument(
    "--reload_models",
    default="False",
    help="Reload models or not(eg. True)")
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

# TODO: Support auto-complete
#argcomplete.autocomplete(parser)

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

# TODO: Check args for model_platform and others

# Initialize flask application
application = Flask(__name__, template_folder='templates')

# Example: {"default": TensorFlowInferenceService}
model_name_service_map = {}

if args.model_config_file != "":
  # Read from configuration file
  with open(args.model_config_file) as data_file:
    model_config_file_dict = json.load(data_file)
    # Example: [{u'platform': u'tensorflow', u'name': u'tensorflow_template_application', u'base_path': u'/Users/tobe/code/simple_tensorflow_serving/models/tensorflow_template_application_model/'}, {u'platform': u'tensorflow', u'name': u'deep_image_model', u'base_path': u'/Users/tobe/code/simple_tensorflow_serving/models/deep_image_model/'}]
    model_config_list = model_config_file_dict["model_config_list"]

    for model_config in model_config_list:
      # Example: {"name": "tensorflow_template_application", "base_path": "/", "platform": "tensorflow"}
      model_name = model_config["name"]
      model_base_path = model_config["base_path"]
      model_platform = model_config.get("platform", "tensorflow")
      custom_op_paths = model_config.get("custom_op_paths", "")


      if model_platform == "tensorflow":
        inference_service = TensorFlowInferenceService(model_name, model_base_path, custom_op_paths, args.verbose)
      if model_platform == "mxnet":
        inference_service = MxnetInferenceService(model_name, model_base_path, args.verbose)

      model_name_service_map[model_name] = inference_service
else:
  # Read from command-line parameter
  if args.model_platform == "tensorflow":
    inference_service = TensorFlowInferenceService(args.model_name, args.model_base_path, args.custom_op_paths, args.verbose)
  elif args.model_platform == "mxnet":
    inference_service = MxnetInferenceService(args.model_name, args.model_base_path, args.verbose)

  model_name_service_map[args.model_name] = inference_service


# Generate client code and exit or not
if args.gen_client != "":
  inference_service = model_name_service_map[args.model_name]
  gen_client.gen_tensorflow_client(inference_service, args.gen_client)
  exit(0)


# Start thread to periodically reload models or not
if args.reload_models == "True" or args.reload_models == "true":
  for model_name, inference_service in model_name_service_map.items():
    if inference_service.platform == "tensorflow":
      inference_service.dynmaically_reload_models()


# The API to render the dashboard page
@application.route("/", methods=["GET"])
@requires_auth
def index():
  return render_template(
      "index.html",
      model_name_service_map=model_name_service_map)


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

  if "model_name" in json_data:
    model_name = json_data.get("model_name", "")
    if model_name == "":
      logging.error("The model does not exist: {}".format(model_name))
  else:
    model_name = "default"

  inferenceService = model_name_service_map[model_name]
  result = inferenceService.inference(json_data)

  # TODO: Change the decoder for numpy data
  return jsonify(json.loads(json.dumps(result, cls=NumpyEncoder)))


def main():
  # Start the HTTP server
  application.run(host=args.host, port=args.port)


if __name__ == "__main__":
  main()
