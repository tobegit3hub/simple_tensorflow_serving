#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import json
import logging
from functools import wraps
from flask import Flask, Response, jsonify, render_template, request
from flask_cors import CORS
import argparse

from manager import InferenceServiceManager
from service_utils import request_util
import python_predict_client
from gen_client import gen_client

logger = logging.getLogger("simple_tensorflow_serving")

# Define parameters
parser = argparse.ArgumentParser()

parser.add_argument(
    "--host",
    default=os.environ.get("STFS_HOST", "0.0.0.0"),
    help="The host of the server(eg. 0.0.0.0)")
parser.add_argument(
    "--port",
    default=int(os.environ.get("STFS_PORT", "8500")),
    help="The port of the server(eg. 8500)",
    type=int)
parser.add_argument(
    "--enable_ssl",
    default=bool(os.environ.get("STFS_ENABLE_SSL", "")),
    help="If enable RESTfull API over https")
parser.add_argument(
    "--secret_pem",
    default=os.environ.get("STFS_SECRET_PEM", "secret.pem"),
    help="SSL pem file")
parser.add_argument(
    "--secret_key",
    default=os.environ.get("STFS_SECRET_KEY", "secret.key"),
    help="SSL key file")
parser.add_argument(
    "--model_name",
    default=os.environ.get("STFS_MODEL_NAME", "default"),
    help="The name of the model(eg. default)")
parser.add_argument(
    "--model_base_path",
    default=os.environ.get("STFS_MODEL_BASE_PATH", "./model"),
    help="The file path of the model(eg. 8500)")
parser.add_argument(
    "--model_platform",
    default=os.environ.get("STFS_MODEL_PLATFORM", "tensorflow"),
    help="The platform of model(eg. tensorflow)")
parser.add_argument(
    "--model_config_file",
    default=os.environ.get("STFS_MODEL_CONFIG_FILE", ""),
    help="The file of the model config(eg. '')")
# TODO: type=bool not works, make it true by default if fixing exit bug
parser.add_argument(
    "--reload_models",
    default=os.environ.get("STFS_RELOAD_MODELS", ""),
    help="Reload models or not(eg. True)")
parser.add_argument(
    "--custom_op_paths",
    default=os.environ.get("STFS_CUSTOM_OP_PATHS", ""),
    help="The path of custom op so files(eg. ./)")
parser.add_argument(
    "--session_config",
    default=os.environ.get("STFS_SESSION_CONFIG", "{}"),
    help="The json of session config")
parser.add_argument(
    "--debug",
    default=os.environ.get("STFS_DEBUG", ""),
    help="Enable debug for flask or not(eg. False)",
    type=bool)
parser.add_argument(
    "--log_level",
    default=os.environ.get("STFS_LOG_LEVEL", "info"),
    help="The log level(eg. info)")
parser.add_argument(
    "--enable_auth",
    default=os.environ.get("STFS_ENABLE_AUTH", ""),
    help="Enable basic auth or not(eg. False)",
    type=bool)
parser.add_argument(
    "--auth_username",
    default=os.environ.get("STFS_AUTH_USERNAME", "admin"),
    help="The username of basic auth(eg. admin)")
parser.add_argument(
    "--auth_password",
    default=os.environ.get("STFS_AUTH_PASSWORD", "admin"),
    help="The password of basic auth(eg. admin)")
parser.add_argument(
    "--enable_cors",
    default=os.environ.get("STFS_ENABLE_CORS", "True"),
    help="Enable cors(eg. True)",
    type=bool)
parser.add_argument(
    "--enable_b64_autoconvert",
    default=os.environ.get("STFS_B64_AUTOCONVERT", ""),
    help="Enable auto convert b64 string(eg. False)",
    type=bool)
parser.add_argument(
    "--download_inference_images",
    default=os.environ.get("STFS_DOWNLOAD_INFERENCE_IMAGES", "True"),
    help="Download inference images(eg. True)",
    type=bool)

# Parse and check parameters
args = parser.parse_args(sys.argv[1:])

for arg in vars(args):
  logger.info("{}: {}".format(arg, getattr(args, arg)))

if args.log_level == "info" or args.log_level == "INFO":
  logger.setLevel(logging.INFO)
elif args.log_level == "debug" or args.log_level == "DEBUG":
  logger.setLevel(logging.DEBUG)
elif args.log_level == "error" or args.log_level == "ERROR":
  logger.setLevel(logging.ERROR)
elif args.log_level == "warning" or args.log_level == "WARNING":
  logger.setLevel(logging.WARNING)
elif args.log_level == "critical" or args.log_level == "CRITICAL":
  logger.setLevel(logging.CRITICAL)

if args.debug == True:
  logger.setLevel(logging.DEBUG)


class WsgiApp:
  """
  The class has Flask app to run by Flask server or uwsgi server.
  """

  def __init__(self, args):
    self.args = args
    self.app = Flask("simple_tensorlow_serving", template_folder='templates')
    self.manager = InferenceServiceManager(args)

    # Initialize Flask app with parameters
    self.app.before_first_request(self.manager.init)
    # TODO: Init the manager before first request
    #self.manager.init()

    if self.args.enable_cors:
      CORS(self.app)

    UPLOAD_FOLDER = os.path.basename('static')
    self.app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    if (os.path.isdir(UPLOAD_FOLDER)):
      pass
    else:
      logging.warn(
          "Create path to host static files: {}".format(UPLOAD_FOLDER))
      os.mkdir(UPLOAD_FOLDER)

    # The API to render the dashboard page
    @self.app.route("/", methods=["GET"])
    @self.requires_auth
    def index():
      return render_template(
          "index.html",
          model_name_service_map=self.manager.model_name_service_map)

    # The API to process inference request
    @self.app.route("/", methods=["POST"])
    @self.requires_auth
    def inference():
      json_result, status_code = self.do_inference()
      response = jsonify(json_result)
      response.status_code = status_code
      return response

    @self.app.route('/health', methods=["GET"])
    def health():
      return Response("healthy")

    @self.app.route('/image_inference', methods=["GET"])
    def image_inference():
      return render_template('image_inference.html')

    @self.app.route('/run_image_inference', methods=['POST'])
    def run_image_inference():
      predict_result = self.do_inference(
          save_file_dir=self.app.config['UPLOAD_FOLDER'])
      json_result = json.dumps(predict_result)

      file = request.files['image']
      image_file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                     file.filename)

      return render_template(
          'image_inference.html',
          image_file_path=image_file_path,
          predict_result=json_result)

    @self.app.route('/json_inference', methods=["GET"])
    def json_inference():
      return render_template('json_inference.html')

    @self.app.route('/run_json_inference', methods=['POST'])
    def run_json_inference():
      # TODO: Fail to parse u'{\r\n  "inputs": [\'\\n\\x1f\\n\\x0e\\n\\x01a\\x12\\t\\n\\x07\\n\\x05hello\\n\\r\\n\\x01b\\x12\\x08\\x12\\x06\\n\\x04\\x00\\x00\\x00?\']\r\n}\r\n          '
      # {
      # "inputs": ['\n\x1f\n\x0e\n\x01a\x12\t\n\x07\n\x05hello\n\r\n\x01b\x12\x08\x12\x06\n\x04\x00\x00\x00?']
      #}
      json_data_string = request.form["json_data"]
      json_data = json.loads(json_data_string)
      model_name = request.form["model_name"]
      model_version = request.form["model_version"]
      signature_name = request.form["signature_name"]

      request_json_data = {
          "model_name": model_name,
          "model_version": model_version,
          "signature_name": signature_name,
          "data": json_data
      }

      predict_result = python_predict_client.predict_json(
          request_json_data, port=args.port)

      return render_template(
          'json_inference.html', predict_result=predict_result)

    # The API to get all models
    @self.app.route("/v1/models", methods=["GET"])
    @self.requires_auth
    def get_models():
      result = [
          inference_service.get_detail() for inference_service in self.manager.
          model_name_service_map.values()
      ]
      return json.dumps(result)

    # The API to get default of the model
    @self.app.route("/v1/models/<model_name>", methods=["GET"])
    @self.requires_auth
    def get_model_detail(model_name):

      if model_name not in self.manager.model_name_service_map:
        return "Model not found: {}".format(model_name)

      inference_service = self.manager.model_name_service_map[model_name]
      return json.dumps(inference_service.get_detail())

    # The API to get example json for client
    @self.app.route("/v1/models/<model_name>/gen_json", methods=["GET"])
    @self.requires_auth
    def gen_example_json(model_name):
      inference_service = self.model_name_service_map[model_name]
      data_json_dict = gen_client.gen_tensorflow_client(
          inference_service, "json", model_name)
      return json.dumps(data_json_dict)

    # The API to get example json for client
    @self.app.route("/v1/models/<model_name>/gen_client", methods=["GET"])
    @self.requires_auth
    def gen_example_client(model_name):
      client_type = request.args.get("language", default="bash", type=str)
      inference_service = self.manager.model_name_service_map[model_name]
      example_client_string = gen_client.gen_tensorflow_client(
          inference_service, client_type, model_name)

      return example_client_string

    @self.app.route("/generate_clients", methods=["GET"])
    def generate_clients():
      return render_template('generate_clients.html')

    @self.app.route("/run_generate_clients", methods=['POST'])
    def run_generate_clients():
      model_name = request.form["model_name"]
      signature_name = request.form["signature_name"]
      language = request.form["language"]

      result = python_predict_client.get_gen_json_and_clients(
          model_name, signature_name, language, port=args.port)

      return render_template("generate_clients.html", result=result)

  def do_inference(self):
    # 1. Check request data format
    if request.content_type.startswith("application/json"):
      # Process requests with json data
      try:
        json_data = request.json
        if not isinstance(json_data, dict):
          result = {"error": "Invalid json data: {}".format(request.data)}
          return result, 400
      except Exception as e:
        result = {"error": "Invalid json data: {}".format(request.data)}
        return result, 400

    elif request.content_type.startswith("multipart/form-data"):
      # Process requests with raw image
      try:
        json_data = request_util.create_json_from_formdata_request(
            request,
            self.args.download_inference_images,
            save_file_dir=self.app.config['UPLOAD_FOLDER'])
      except Exception as e:
        result = {"error": "Invalid form-data: {}".format(e)}
        return result, 400

    else:
      logging.error("Unsupported content type: {}".format(
          request.content_type))
      return {"error": "Error, unsupported content type"}, 400

    # 2. Get model or use default one
    model_name = "default"
    if "model_name" in json_data:
      model_name = json_data.get("model_name")

    if model_name not in self.manager.model_name_service_map:
      return {
          "error":
          "Invalid model name: {}, available models: {}".format(
              model_name, self.model_name_service_map.keys())
      }, 400

    # 3. Use initialized manager for inference
    try:
      result = self.manager.inference(model_name, json_data)
      return result, 200
    except Exception as e:
      result = {"error": e.message}
      return result, 400

  def verify_authentication(self, username, password):
    """
      Verify if this user should be authenticated or not.
    
      Args:
        username: The user name.
        password: The password.
    
      Return:
        True if it passes and False if it does not pass.
      """
    if self.args.enable_auth:
      if username == self.args.auth_username and password == self.args.auth_password:
        return True
      else:
        return False
    else:
      return True

  def requires_auth(self, f):
    """
      The decorator to enable basic auth.
      """

    @wraps(f)
    def decorated(*decorator_args, **decorator_kwargs):

      auth = request.authorization

      if self.args.enable_auth:
        if not auth or not self.verify_authentication(auth.username,
                                                      auth.password):
          response = Response(
              "Need basic auth to request the resources", 401,
              {"WWW-Authenticate": '"Basic realm="Login Required"'})
          return response

      return f(*decorator_args, **decorator_kwargs)

    return decorated


app = WsgiApp(args).app


def main():
  # Run with Flask HTTP server
  if args.enable_ssl:
    # Can pass ssl_context="adhoc" for auto-sign certification
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=args.debug,
        ssl_context=(args.secret_pem, args.secret_key))
  else:
    app.run(host=args.host, port=args.port, threaded=True, debug=args.debug)


if __name__ == "__main__":
  main()
