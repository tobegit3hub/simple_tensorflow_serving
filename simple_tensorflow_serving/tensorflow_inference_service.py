import logging
import os
import time

import tensorflow as tf

from abstract_inference_service import AbstractInferenceService


class TensorFlowInferenceService(AbstractInferenceService):
  """
  The TensorFlow service to load TensorFlow SavedModel and make inference.
  """

  def __init__(self, model_base_path, model_name, model_version,
               verbose=False):
    """
    Initialize the TensorFlow service by loading SavedModel to the Session.
        
    Args:
      model_base_path: The file path of the model.
      model_name: The name of the model.
      model_version: The version of the model.
    Return:
    """

    self.sess = tf.Session(graph=tf.Graph())
    self.meta_graph = self.load_savedmodel(self.sess, model_base_path,
                                           model_name, model_version)
    self.graph_signature = self.meta_graph.signature_def.items()[0][1]
    self.verbose = verbose

  def load_savedmodel(self, sess, model_base_path, model_name, model_version):
    """
    Load the SavedModel, update the Session object and return the Graph object.

    Args:
      sess: The Session object to restore.
      model_base_path: The file path of the model.
      model_name: The name of the model.
      model_version: The version of the model.
    Return:
      The meta Graph object.
    """

    model_file_path = os.path.join(model_base_path, str(model_version))
    logging.info("Load the TensorFlow model: {}, version: {}, path: {}".format(
        model_name, model_version, model_file_path))

    meta_graph = tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], model_file_path)
    return meta_graph

  def inference(self, input_data):
    """
    Make inference with the current Session object and JSON request data.
        
    Args:
      input_data: The JSON serialized object with key and array data.
                  Example is {"keys": [[1.0], [2.0]], "features": [[10, 10, 10, 8, 6, 1, 8, 9, 1], [6, 2, 1, 1, 1, 1, 7, 1, 1]]}.
    Return:
      The JSON serialized object with key and array data.
      Example is {"keys": [[11], [2]], "softmax": [[0.61554497, 0.38445505], [0.61554497, 0.38445505]], "prediction": [0, 0]}.
    """

    if self.verbose:
      logging.debug("Inference data: {}".format(input_data))

    # 1. Build feed dict for input data
    feed_dict_map = {}
    for input_item in self.graph_signature.inputs.items():
      # Example: "keys"
      input_op_name = input_item[0]
      # Example: "Placeholder_0"
      input_tensor_name = input_item[1].name
      # Example: {"Placeholder_0": [[1.0], [2.0]], "Placeholder_1:0": [[10, 10, 10, 8, 6, 1, 8, 9, 1], [6, 2, 1, 1, 1, 1, 7, 1, 1]]}
      feed_dict_map[input_tensor_name] = input_data[input_op_name]

    # 2. Build inference operators
    output_tensor_names = []
    output_op_names = []
    for output_item in self.graph_signature.outputs.items():
      # Example: "keys"
      output_op_name = output_item[0]
      output_op_names.append(output_op_name)
      # Example: "Identity:0"
      output_tensor_name = output_item[1].name
      output_tensor_names.append(output_tensor_name)

    # 3. Inference with Session run
    if self.verbose:
      start_time = time.time()
    result_ndarrays = self.sess.run(
        output_tensor_names, feed_dict=feed_dict_map)
    if self.verbose:
      logging.debug("Inference time: {} s".format(time.time() - start_time))

    # 4. Build return result
    result = {}
    for i in range(len(output_op_names)):
      result[output_op_names[i]] = result_ndarrays[i]
    if self.verbose:
      logging.debug("Inference result: {}".format(result))
    return result
