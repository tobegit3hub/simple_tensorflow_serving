import logging
import os
import signal
import threading
import time

import tensorflow as tf

from abstract_inference_service import AbstractInferenceService


class TensorFlowInferenceService(AbstractInferenceService):
  """
  The TensorFlow service to load TensorFlow SavedModel and make inference.
  """

  def __init__(self, model_base_path, verbose=False):
    """
    Initialize the TensorFlow service by loading SavedModel to the Session.
        
    Args:
      model_base_path: The file path of the model.
      model_name: The name of the model.
      model_version: The version of the model.
    Return:
    """

    self.model_base_path = model_base_path
    self.model_versions = []
    self.sessions = []
    self.meta_graph = None
    self.graph_signature = None
    self.verbose = verbose
    self.should_stop_all_threads = False

    # Register signal to exist
    signal.signal(signal.SIGTERM, self.stop_all_threads)
    signal.signal(signal.SIGINT, self.stop_all_threads)

    # Start new thread to load models
    load_savedmodels_thread = threading.Thread(
        target=self.load_savedmodels_thread, args=())
    load_savedmodels_thread.start()
    # dynamically_load_savedmodels_thread.join()

  def stop_all_threads(self, signum, frame):
    logging.info("Catch signal {} and exit all threads".format(signum))
    self.should_stop_all_threads = True
    exit(0)

  def load_savedmodels_thread(self):
    """
    Load the SavedModel, update the Session object and return the Graph object.
    """

    while self.should_stop_all_threads == False:
      # TODO: Add lock if needed
      current_model_versions = os.listdir(self.model_base_path)

      if current_model_versions == self.model_versions:
        # No version change, just sleep
        if self.verbose:
          logging.debug("Watch the model path: {} and sleep {} seconds".format(
              self.model_base_path, 10))
        time.sleep(10)

      else:
        # Versions change, reload all the SavedModel
        self.model_versions = current_model_versions
        logging.info("Detect models change, reload the model versions: {}".
                     format(self.model_versions))
        del self.sessions
        self.sessions = []

        for model_version in self.model_versions:
          # TODO: Just re-load the changed model versions
          session = tf.Session(graph=tf.Graph())
          self.sessions.append(session)

          model_file_path = os.path.join(self.model_base_path, model_version)
          logging.info("Load the TensorFlow model version: {}, path: {}".
                       format(model_version, model_file_path))
          self.meta_graph = tf.saved_model.loader.load(
              session, [tf.saved_model.tag_constants.SERVING], model_file_path)
          self.graph_signature = self.meta_graph.signature_def.items()[0][1]

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
    # TODO: Optimize to pre-compute this before inference
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
    result_ndarrays = self.sessions[0].run(
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
