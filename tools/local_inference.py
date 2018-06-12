#!/usr/bin/env python

import os
import time
import tensorflow as tf
import logging

logging.basicConfig(level=logging.DEBUG)


class LocalInferenceService(object):
  def __init__(self, model_base_path, model_version):

    self.model_base_path = model_base_path
    self.model_version = model_version
    self.model_graph_signature = None
    self.session = tf.Session(graph=tf.Graph())

    self.load_model()

  def load_model(self):

    model_file_path = os.path.join(self.model_base_path,
                                   str(self.model_version))
    logging.info("Try to load the model in: {}".format(model_file_path))

    try:
      meta_graph = tf.saved_model.loader.load(self.session, [
          tf.saved_model.tag_constants.SERVING
      ], model_file_path)
      logging.info("Succeed to load model in: {}".format(model_file_path))

      self.model_graph_signature = list(meta_graph.signature_def.items())[0][1]

    except IOError as ioe:
      logging.info("Fail to load model and catch exception: {}".format(ioe))

  def inference(self, json_data):
    # 1. Build feed dict for input data
    feed_dict_map = {}
    input_data = json_data.get("data", "")

    for input_item in self.model_graph_signature.inputs.items():
      input_op_name = input_item[0]
      input_tensor_name = input_item[1].name
      feed_dict_map[input_tensor_name] = input_data[input_op_name]

    # 2. Build inference operators
    output_tensor_names = []
    output_op_names = []
    for output_item in self.model_graph_signature.outputs.items():

      if output_item[1].name != "":
        output_op_name = output_item[0]
        output_op_names.append(output_op_name)
        output_tensor_name = output_item[1].name
        output_tensor_names.append(output_tensor_name)
      elif output_item[1].coo_sparse != None:
        # For SparseTensor op, Example: values_tensor_name: "CTCBeamSearchDecoder_1:1", indices_tensor_name: "CTCBeamSearchDecoder_1:0", dense_shape_tensor_name: "CTCBeamSearchDecoder_1:2"
        values_tensor_name = output_item[1].coo_sparse.values_tensor_name
        indices_tensor_name = output_item[1].coo_sparse.indices_tensor_name
        dense_shape_tensor_name = output_item[
            1].coo_sparse.dense_shape_tensor_name
        output_op_names.append("{}_{}".format(output_item[0], "values"))
        output_op_names.append("{}_{}".format(output_item[0], "indices"))
        output_op_names.append("{}_{}".format(output_item[0], "shape"))
        output_tensor_names.append(values_tensor_name)
        output_tensor_names.append(indices_tensor_name)
        output_tensor_names.append(dense_shape_tensor_name)

    # 3. Inference with Session run
    start_time = time.time()
    result_ndarrays = self.session.run(
        output_tensor_names, feed_dict=feed_dict_map)
    logging.debug("Inference time: {} s".format(time.time() - start_time))

    # 4. Build return result
    result = {}
    for i in range(len(output_op_names)):
      result[output_op_names[i]] = result_ndarrays[i]
    logging.debug("Inference result: {}".format(result))
    return result


def main():

  model_base_path = "../models/tensorflow_template_application_model/"
  model_version = "1"
  json_data = {
      "data": {
          "keys": [[1.0], [2.0]],
          "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1]]
      }
  }

  service = LocalInferenceService(model_base_path, model_version)
  service.inference(json_data)


if __name__ == "__main__":
  main()
