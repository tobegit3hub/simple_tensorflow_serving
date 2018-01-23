#!/usr/bin/env python

import json
import logging
import pprint

import tensorflow as tf
from flask import Flask, request

# Define parameters
flags = tf.app.flags
flags.DEFINE_boolean("enable_colored_log", True, "Enable colored log")
flags.DEFINE_string("host", "0.0.0.0", "The host of the server")
flags.DEFINE_integer("port", 8500, "The port of the server")
flags.DEFINE_string("model_name", "default", "The name of the model")
flags.DEFINE_integer("model_version", 1, "The version of the model")
flags.DEFINE_string("model_base_path", "./model", "The file path of the model")
FLAGS = flags.FLAGS

logging.basicConfig(level=logging.DEBUG)
if FLAGS.enable_colored_log:
  import coloredlogs
  coloredlogs.install()
pprint.PrettyPrinter().pprint(FLAGS.__flags)


class TensorFlowService(object):
  def __init__(self):
    self.model = ""

    self.sess = tf.Session(graph=tf.Graph())
    self.meta_graph = tf.saved_model.loader.load(self.sess, [
        tf.saved_model.tag_constants.SERVING
    ], FLAGS.model_base_path)

  def inference(self, input_data):

    item = self.meta_graph.signature_def.items()[0][1]

    #request_input_data = {"keys": [[1.0], [2.0]], "features": [[10, 10, 10, 8, 6, 1, 8, 9, 1], [6, 2, 1, 1, 1, 1, 7, 1, 1]]}

    feed_dict_map = {}
    for input_item in item.inputs.items():
      # Example: "keys"
      input_op_name = input_item[0]
      # Example: "Placeholder_1:0"
      input_tensor_name = input_item[1].name
      # {input1_name: [[1.0], [2.0]], input2_name: [[10, 10, 10, 8, 6, 1, 8, 9, 1], [6, 2, 1, 1, 1, 1, 7, 1, 1]]}
      feed_dict_map[input_tensor_name] = input_data[input_op_name]

    #output_names = [output1_name, output2_name]
    output_tensor_names = []
    output_op_names = []

    for output_item in item.outputs.items():
      # Example: "keys"
      output_op_name = output_item[0]
      # Example: "Identity:0"
      output_op_names.append(output_op_name)
      output_tensor_name = output_item[1].name
      output_tensor_names.append(output_tensor_name)

    result_ndarrays = self.sess.run(
        output_tensor_names, feed_dict=feed_dict_map)
    result = {}
    for i in range(len(output_op_names)):
      result[output_op_names[i]] = result_ndarrays[i]
    pprint.PrettyPrinter().pprint(result)
    return result


def main():

  app = Flask(__name__)

  tensorflowService = TensorFlowService()

  @app.route("/", methods=["GET"])
  def index():
    return "Get is not supported"

  @app.route("/", methods=["POST"])
  def inference():
    data = json.loads(request.data)
    result = tensorflowService.inference(data)
    return str(result)

  app.run(host=FLAGS.host, port=FLAGS.port)


if __name__ == "__main__":
  main()
