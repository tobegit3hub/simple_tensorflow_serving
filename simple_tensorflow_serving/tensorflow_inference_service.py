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

  def __init__(self, model_name, model_base_path, custom_op_paths="", verbose=False):
    """
    Initialize the TensorFlow service by loading SavedModel to the Session.
        
    Args:
      model_name: The name of the model.
      model_base_path: The file path of the model.
    Return:
      None
    """

    super(TensorFlowInferenceService, self).__init__()

    self.model_name = model_name
    self.model_base_path = model_base_path
    self.model_version_list = []
    self.model_graph_signature = None
    self.platform = "TensorFlow"

    if custom_op_paths != "":
      self.load_custom_op(custom_op_paths)

    self.version_session_map = {}

    self.verbose = verbose
    self.should_stop_all_threads = False

    # Register the signals to exist
    signal.signal(signal.SIGTERM, self.stop_all_threads)
    signal.signal(signal.SIGINT, self.stop_all_threads)

    model_version = self.get_one_model_version()
    self.load_saved_model_version(model_version)

  def load_custom_op(self, custom_op_paths):

    custom_op_path_list = custom_op_paths.split(",")

    for custom_op_path in custom_op_path_list:
      if os.path.isdir(custom_op_path):
        for filename in os.listdir(custom_op_path):
          if filename.endswith(".so"):

            op_filepath = os.path.join(custom_op_path, filename)
            logging.info("Load the so file from: {}".format(op_filepath))
            tf.load_op_library(op_filepath)

      else:
        logging.error("The path does not exist: {}".format(custom_op_path))

  def dynmaically_reload_models(self):
    """
    Start new thread to load models periodically.

    Return:
      None
    """

    logging.info("Start the new thread to periodically reload model versions")
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

    Return:
      None
    """

    while self.should_stop_all_threads == False:
      # TODO: Add lock if needed
      current_model_versions_string = os.listdir(self.model_base_path)
      current_model_versions = set([
          int(version_string)
          for version_string in current_model_versions_string
      ])

      old_model_versions_string = self.version_session_map.keys()
      old_model_versions = set([
          int(version_string) for version_string in old_model_versions_string
      ])

      if current_model_versions == old_model_versions:
        # No version change, just sleep
        if self.verbose:
          logging.debug("Watch the model path: {} and sleep {} seconds".format(
              self.model_base_path, 10))
        time.sleep(10)

      else:
        # Versions change, load the new models and offline the deprecated ones
        logging.info(
            "Model path updated, change model versions from: {} to: {}".format(
                old_model_versions, current_model_versions))

        # Put old model versions offline
        offline_model_versions = old_model_versions - current_model_versions
        for model_version in offline_model_versions:
          logging.info(
              "Put the model version: {} offline".format(str(model_version)))
          del self.version_session_map[str(model_version)]
          self.version_session_map.remove(model_version)


      # Create Session for new model version
        online_model_versions = current_model_versions - old_model_versions
        for model_version in online_model_versions:
          self.load_saved_model_version(model_version)

  def load_saved_model_version(self, model_version):
    session = tf.Session(graph=tf.Graph())
    self.version_session_map[str(model_version)] = session
    self.model_version_list.append(model_version)

    model_file_path = os.path.join(self.model_base_path, str(model_version))
    logging.info("Put the model version: {} online, path: {}".format(
        model_version, model_file_path))
    meta_graph = tf.saved_model.loader.load(
        session, [tf.saved_model.tag_constants.SERVING], model_file_path)
    self.model_graph_signature = meta_graph.signature_def.items()[0][1]

  def get_one_model_version(self):
    current_model_versions_string = os.listdir(self.model_base_path)
    if len(current_model_versions_string) > 0:
      return int(current_model_versions_string[0])
    else:
      logging.error("No model version found")

  # TODO: load all model version by default
  #def load_all_saved_model_version(self):


  def inference(self, json_data):
    """
    Make inference with the current Session object and JSON request data.
        
    Args:
      json_data: The JSON serialized object with key and array data.
                 Example is {"model_version": 1, "data": {"keys": [[1.0], [2.0]], "features": [[10, 10, 10, 8, 6, 1, 8, 9, 1], [6, 2, 1, 1, 1, 1, 7, 1, 1]]}}.
    Return:
      The dictionary with key and array data.
      Example is {"keys": [[11], [2]], "softmax": [[0.61554497, 0.38445505], [0.61554497, 0.38445505]], "prediction": [0, 0]}.
    """

    # Use the latest model version if not specified
    model_version = int(json_data.get("model_version", -1))
    input_data = json_data.get("data", "")
    if model_version == -1:
      for model_version_string in self.version_session_map.keys():
        if int(model_version_string) > model_version:
          model_version = int(model_version_string)

    if str(model_version) not in self.version_session_map or input_data == "":
      logging.error("No model version: {} to serve".format(model_version))
      return "Fail to request the model version: {} with data: {}".format(
          model_version, input_data)

    if self.verbose:
      logging.debug("Inference model_version: {}, data: {}".format(
          model_version, input_data))

    # 1. Build feed dict for input data
    feed_dict_map = {}
    for input_item in self.model_graph_signature.inputs.items():
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
    for output_item in self.model_graph_signature.outputs.items():
      # Example: "keys"
      output_op_name = output_item[0]
      output_op_names.append(output_op_name)
      # Example: "Identity:0"
      output_tensor_name = output_item[1].name
      output_tensor_names.append(output_tensor_name)

    # 3. Inference with Session run
    if self.verbose:
      start_time = time.time()
    sess = self.version_session_map[str(model_version)]
    result_ndarrays = sess.run(output_tensor_names, feed_dict=feed_dict_map)
    if self.verbose:
      logging.debug("Inference time: {} s".format(time.time() - start_time))

    # 4. Build return result
    result = {}
    for i in range(len(output_op_names)):
      result[output_op_names[i]] = result_ndarrays[i]
    if self.verbose:
      logging.debug("Inference result: {}".format(result))
    return result
