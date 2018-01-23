#!/usr/bin/env python

import json

import tensorflow as tf
from flask import Flask, request

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
  return "Test endpoint"


@app.route("/", methods=["POST"])
def main():
  # Predict request
  data = json.loads(request.data)

  result = []
  for predict_sample in data:
    response = service.process_request(predict_sample)
    result.append(response)

  return str(result)


if __name__ == "__main__":

  export_dir = "../model/1/"

  with tf.Session(graph=tf.Graph()) as sess:
    meta_graph = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    """
    inputs {
      key: "features"
      value {
        name: "Placeholder:0"
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: -1
          }
          dim {
            size: 9
          }
        }
      }
    }
    inputs {
      key: "keys"
      value {
        name: "Placeholder_1:0"
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: -1
          }
          dim {
            size: 1
          }
        }
      }
    }
    outputs {
      key: "keys"
      value {
        name: "Identity:0"
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: -1
          }
          dim {
            size: 1
          }
        }
      }
    }
    outputs {
      key: "prediction"
      value {
        name: "ArgMax_2:0"
        dtype: DT_INT64
        tensor_shape {
          dim {
            size: -1
          }
        }
      }
    }
    outputs {
      key: "softmax"
      value {
        name: "Softmax_2:0"
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: -1
          }
          dim {
            size: 2
          }
        }
      }
    }
    method_name: "tensorflow/serving/classify"
    """
    item = meta_graph.signature_def.items()[0][1]

    input1 = item.inputs.items()[0]
    input1_name = input1[0]
    input1_name = "Placeholder_1:0"
    input2 = item.inputs.items()[1]
    input2_name = input2[0]
    input2_name = "Placeholder:0"

    output1 = item.outputs.items()[0]
    output1_name = output1[0]
    output1_name = "Identity:0"
    output2 = item.outputs.items()[1]
    output2_name = output2[0]
    output2_name = "Softmax_2:0"

    output_names = [output1_name, output2_name]

    result = sess.run(
        output_names,
        feed_dict={
            input1_name: [[1.0], [2.0]],
            input2_name: [[10, 10, 10, 8, 6, 1, 8, 9, 1],
                          [6, 2, 1, 1, 1, 1, 7, 1, 1]]
        })
    print(result)

  app.run(host="0.0.0.0")
