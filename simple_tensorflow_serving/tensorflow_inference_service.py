# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import signal
import threading
import time
import base64
import tensorflow as tf
import marshal
import types

from .abstract_inference_service import AbstractInferenceService
from . import filesystem_util

logger = logging.getLogger('simple_tensorflow_serving')


class TensorFlowInferenceService(AbstractInferenceService):
  """
  The TensorFlow service to load TensorFlow SavedModel and make inference.
  """

  def __init__(self,
               model_name,
               model_base_path,
               custom_op_paths="",
               session_config={}):
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
    libhdfs_model_base_path = filesystem_util.update_hdfs_prefix_for_libhdfs(
        model_base_path)
    self.model_base_path = libhdfs_model_base_path
    self.model_version_list = []
    self.model_graph_signature = None
    self.platform = "TensorFlow"
    self.session_config = session_config

    self.name_signature_map = {}
    self.preprocess_function = None
    self.postprocess_function = None

    if custom_op_paths != "":
      self.load_custom_op(custom_op_paths)

    self.version_session_map = {}
    self.profiler_map = {}

    self.should_stop_all_threads = False

    # Register the signals to exist
    signal.signal(signal.SIGTERM, self.stop_all_threads)
    signal.signal(signal.SIGINT, self.stop_all_threads)

    model_versions = self.get_all_model_versions()
    for model_version in model_versions:
      self.load_saved_model_version(model_version)

  def load_custom_op(self, custom_op_paths):

    custom_op_path_list = custom_op_paths.split(",")

    for custom_op_path in custom_op_path_list:
      if os.path.isdir(custom_op_path):
        for filename in os.listdir(custom_op_path):
          if filename.endswith(".so"):

            op_filepath = os.path.join(custom_op_path, filename)
            logger.info("Load the so file from: {}".format(op_filepath))
            tf.load_op_library(op_filepath)

      else:
        logger.error("The path does not exist: {}".format(custom_op_path))

  def dynamically_reload_models(self):
    """
    Start new thread to load models periodically.

    Return:
      None
    """

    logger.info("Start the new thread to periodically reload model versions")
    load_savedmodels_thread = threading.Thread(
        target=self.load_savedmodels_thread, args=())
    load_savedmodels_thread.start()
    # dynamically_load_savedmodels_thread.join()

  def stop_all_threads(self, signum, frame):
    logger.info("Catch signal {} and exit all threads".format(signum))
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
      # TODO: Support HDFS with TensorFlow API
      current_model_versions_string = os.listdir(self.model_base_path)
      current_model_versions = set(
          [version_string for version_string in current_model_versions_string])

      old_model_versions_string = self.version_session_map.keys()
      old_model_versions = set(
          [version_string for version_string in old_model_versions_string])

      if current_model_versions == old_model_versions:
        # No version change, just sleep
        logger.debug("Watch the model path: {} and sleep {} seconds".format(
            self.model_base_path, 10))
        time.sleep(10)

      else:
        # Versions change, load the new models and offline the deprecated ones
        logger.info(
            "Model path updated, change model versions from: {} to: {}".format(
                old_model_versions, current_model_versions))

        # Put old model versions offline
        offline_model_versions = old_model_versions - current_model_versions
        for model_version in offline_model_versions:
          logger.info(
              "Put the model version: {} offline".format(str(model_version)))
          del self.version_session_map[str(model_version)]
          self.version_session_map.remove(model_version)

      # Create Session for new model version
        online_model_versions = current_model_versions - old_model_versions
        for model_version in online_model_versions:
          self.load_saved_model_version(model_version)

  def load_saved_model_version(self, model_version):

    config = tf.ConfigProto()
    if "log_device_placement" in self.session_config:
      config.log_device_placement = self.session_config["log_device_placement"]
    if "allow_soft_placement" in self.session_config:
      config.allow_soft_placement = self.session_config["allow_soft_placement"]
    if "allow_growth" in self.session_config:
      config.gpu_options.allow_growth = self.session_config["allow_growth"]
    if "per_process_gpu_memory_fraction" in self.session_config:
      config.gpu_options.per_process_gpu_memory_fraction = self.session_config[
          "per_process_gpu_memory_fraction"]

    session = tf.Session(graph=tf.Graph(), config=config)

    self.version_session_map[str(model_version)] = session
    self.model_version_list.append(model_version)

    model_file_path = os.path.join(self.model_base_path, str(model_version))
    logger.info("Put the model version: {} online, path: {}".format(
        model_version, model_file_path))
    meta_graph = tf.saved_model.loader.load(
        session, [tf.saved_model.tag_constants.SERVING], model_file_path)

    # Get preprocess and postprocess function from collection_def
    if "preprocess_function" in meta_graph.collection_def:
      logging.info("Load the preprocess function in graph")
      preprocess_function_string = meta_graph.collection_def[
          "preprocess_function"].bytes_list.value[0]
      loaded_function = marshal.loads(preprocess_function_string)
      self.preprocess_function = types.FunctionType(loaded_function,
                                                    globals(),
                                                    "preprocess_function")

    if "postprocess_function" in meta_graph.collection_def:
      logging.info("Load the postprocess function in graph")
      postrocess_function_string = meta_graph.collection_def[
          "postprocess_function"].bytes_list.value[0]
      loaded_function = marshal.loads(postrocess_function_string)
      self.postprocess_function = types.FunctionType(loaded_function,
                                                     globals(),
                                                     "postprocess_function")

    # Update ItemsView to list for Python 3
    if len(list(meta_graph.signature_def.items())) == 1:
      signature_name = list(meta_graph.signature_def.items())[0][0]
      self.model_graph_signature = list(meta_graph.signature_def.items())[0][1]
      self.model_graph_signature_dict = tensorflow_model_graph_to_dict(
          self.model_graph_signature)
      self.name_signature_map[signature_name] = self.model_graph_signature
    else:
      index = 0
      for item in list(meta_graph.signature_def.items()):
        signature_name = item[0]
        self.name_signature_map[signature_name] = item[1]

        #if signature_name == tf.python.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
        if signature_name == "serving_default":
          self.model_graph_signature = item[1]
          self.model_graph_signature_dict = tensorflow_model_graph_to_dict(
              self.model_graph_signature)
        elif self.model_graph_signature == None and index == (
            len(list(meta_graph.signature_def.items())) - 1):
          self.model_graph_signature = item[1]
          self.model_graph_signature_dict = tensorflow_model_graph_to_dict(
              self.model_graph_signature)

        index += 1

  def get_one_model_version(self):
    all_model_versions = self.get_all_model_versions()
    # current_model_versions_string = os.listdir(self.model_base_path)

    if len(all_model_versions) > 0:
      return all_model_versions[0]
    else:
      logger.error("No model version found")

  def get_all_model_versions(self):
    """
    if self.model_base_path.startswith("hdfs://"):
      # TODO: Support getting sub-directory from HDFS
      #return [0]

      # hdfs://172.27.128.31:8020/user/chendihao/tensorflow_template_application/model/
      # hdfs://default/user/chendihao/tensorflow_template_application/model/
      # hdfs:///user/chendihao/tensorflow_template_application/model/
      "
      hadoop_user_name = os.environ.get("HDFS_USER_NAME", "work")
      hadoop_enable_kerberos = os.environ.get("HDFS_ENABLE_KERBEROS", "false")

      if hadoop_enable_kerberos == "true" or hadoop_enable_kerberos == "True":
        hadoop_kerberos_ticket_file = os.environ.get(
            "HDFS_KERBEROS_TICKET_FILE", "/tmp/krb5cc_0")
        hadoop_kinit_command = os.environ.get(
            "HDFS_KINIT_COMMAND",
            "kinit {} -k -t /etc/user.keytab".format(hadoop_user_name))
        logger.info(
            "Try to run the command to kinit: {}".format(hadoop_kinit_command))
        subprocess.call(hadoop_kinit_command, shell=True)

      if self.model_base_path.startswith("hdfs:///"):
        hadoop_endpoint = os.environ.get("HDFS_ENDPOINT", "default")
      else:
        # 'hdfs://172.27.128.31:8020/user/chendihao/tensorflow_template_application/model/' -> ['hdfs:', '', '172.27.128.31:8020', 'user', 'chendihao', 'tensorflow_template_application', 'model', '']
        hadoop_endpoint = self.model_base_path.split("/")[2]

      if len(hadoop_endpoint.split(":")) == 2:
        # 172.27.128.31:8020
        hdfs_host_port_pair = hadoop_endpoint.split(":")
        hdfs_host = hdfs_host_port_pair[0]
        hdfs_port = int(hdfs_host_port_pair[1])
      else:
        # "default" or "hacluster" and so on
        hdfs_host = hadoop_endpoint
        hdfs_port = 0

      
      if hadoop_enable_kerberos == "true" or hadoop_enable_kerberos == "True":
        client = pa.hdfs.connect(
            host=hdfs_host,
            port=hdfs_port,
            user=hadoop_user_name,
            kerb_ticket=hadoop_kerberos_ticket_file)
      else:
        client = pa.hdfs.connect(
            host=hdfs_host, port=hdfs_port, user=hadoop_user_name)

      if self.model_base_path.startswith("hdfs:///"):
        model_version_paths = client.ls(self.model_base_path)
      else:
        if hdfs_host == "default":
          # Remove the "default" for pyarrow
          model_version_paths = client.ls(self.model_base_path[:7] +
                                          self.model_base_path[14:])
        else:
          model_version_paths = client.ls(self.model_base_path)

      model_versions = [
          model_version_path.split("/")[-1]
          for model_version_path in model_version_paths
      ]
      "

      # TODO: Support only integer model version or not
      model_versions = tf.gfile.ListDirectory(self.model_base_path)
      return model_versions

    else:

      current_model_versions_string = os.listdir(self.model_base_path)
      if len(current_model_versions_string) > 0:
        model_versions = [
            int(model_version_string)
            for model_version_string in current_model_versions_string
        ]
        return model_versions
      else:
        logger.error("No model version found")
        return []
    """

    model_versions = tf.gfile.ListDirectory(self.model_base_path)
    return model_versions

  def run_with_profiler(self, session, version, output_tensors, feed_dict):
    if version not in self.profiler_map:
      if len(self.profiler_map) > 0:
        logger.warn(
            "Only support one profiler per process, run without profiler")
        return session.run(output_tensors, feed_dict), None
      profiler = tf.profiler.Profiler(session.graph)
      self.profiler_map[version] = profiler
    else:
      profiler = self.profiler_map[version]
    run_meta = tf.RunMetadata()
    result = session.run(
        output_tensors,
        feed_dict=feed_dict,
        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        run_metadata=run_meta)
    profiler.add_step(0, run_meta)
    profiler_out_file = "/tmp/.simple_tensorflow_serving_prof-" + str(
        int(time.time()))
    opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
    opts["output"] = "file:outfile=%s" % profiler_out_file
    profiler.profile_operations(options=opts)
    profile_result = None
    try:
      with open(profiler_out_file) as f:
        profile_result = f.read()
    except Exception as e:
      logger.error(e.message)
    return result, profile_result

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
    model_version = json_data.get("model_version", "-1")
    input_data = json_data.get("data", "")

    if model_version == "-1":
      # TODO: Make sure it use the latest model version
      model_version = list(self.version_session_map.keys())[-1]

    if str(model_version) not in self.version_session_map or input_data == "":
      logger.error("No model version: {} to serve".format(model_version))
      return "Fail to request the model version: {} with data: {}".format(
          model_version, input_data)

    logger.debug("Inference with json data: {}".format(json_data))

    if json_data.get("preprocess", "false") != "false":
      if self.preprocess_function != None:
        input_data = self.preprocess_function(input_data)
        logger.debug("Preprocess to generate data: {}".format(input_data))
      else:
        logger.warning("No preprocess function in model")

    signature_name = json_data.get("signature_name", "")
    if signature_name == "":
      current_model_graph_signature = self.model_graph_signature
    else:
      if signature_name in self.name_signature_map:
        current_model_graph_signature = self.name_signature_map[signature_name]
      else:
        return "Fail to request the signature name: {}".format(signature_name)

    # 1. Build feed dict for input data
    feed_dict_map = {}
    for input_item in current_model_graph_signature.inputs.items():
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
    for output_item in current_model_graph_signature.outputs.items():

      if output_item[1].name != "":
        # Example: "keys"
        output_op_name = output_item[0]
        output_op_names.append(output_op_name)
        # Example: "Identity:0"
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

    sess = self.version_session_map[str(model_version)]
    result_profile = None
    if json_data.get("run_profile", "") == "true":
      logger.info("run_profile=true, running with tfprof")
      result_ndarrays, result_profile = self.run_with_profiler(
          sess, str(model_version), output_tensor_names, feed_dict_map)
    else:

      # TODO: Update input data by decoding base64 string for esitmator model
      should_decode_base64 = False
      # Should not use have_key for Python 3
      if "input_example_tensor:0" in feed_dict_map and should_decode_base64:
        final_example_strings = []
        base64_example_strings = feed_dict_map["input_example_tensor:0"]
        for base64_example_string in base64_example_strings:
          final_example_string = base64.urlsafe_b64decode(
              base64_example_string.encode("utf-8"))
          final_example_strings.append(final_example_string)
        feed_dict_map["input_example_tensor:0"] = final_example_strings

      result_ndarrays = sess.run(output_tensor_names, feed_dict=feed_dict_map)

    logger.debug("Inference time: {} s".format(time.time() - start_time))

    # 4. Build return result
    result = {}
    for i in range(len(output_op_names)):
      result[output_op_names[i]] = result_ndarrays[i]
    logger.debug("Inference result: {}".format(result))

    if json_data.get("postprocess", "false") != "false":
      if self.postprocess_function != None:
        result = self.postprocess_function(result)
        logger.debug("Postprocess to generate data: {}".format(result))
      else:
        logger.warning("No postprocess function in model")

    # 5. Build extra return information
    if result_profile is not None and "__PROFILE__" not in output_tensor_names:
      result["__PROFILE__"] = result_profile

    return result


def tensorflow_model_graph_to_dict(model_graph_signature):
  model_graph_signature_dict = {}
  model_graph_signature_dict["inputs"] = []
  model_graph_signature_dict["outputs"] = []

  for input_item in model_graph_signature.inputs.items():
    # Example: {"name: "keys", "dtype": 1(DT_INT32), "shape": [-1, 1]}
    input_map = {"name": "", "dtype": 0, "shape": []}

    # Example: "keys"
    input_opname = input_item[0]
    input_map["name"] = input_opname

    dtype = input_item[1].dtype
    input_map["dtype"] = dtype

    # Example: [-1, 1]
    shape_dims = input_item[1].tensor_shape.dim

    for dim in shape_dims:
      input_map["shape"].append(int(dim.size))

    model_graph_signature_dict["inputs"].append(input_map)

  for output_item in model_graph_signature.outputs.items():

    if output_item[1].name != "":
      # Example: {"name: "keys", "dtype": 1(DT_INT32), "shape": [-1, 1]}
      output_map = {"name": "", "dtype": 0, "shape": []}

      # Example: "keys"
      output_op_name = output_item[0]
      output_map["name"] = output_op_name

      dtype = output_item[1].dtype
      output_map["dtype"] = dtype

      # Example: [-1, 1]
      shape_dims = output_item[1].tensor_shape.dim

      for dim in shape_dims:
        output_map["shape"].append(int(dim.size))

      model_graph_signature_dict["outputs"].append(output_map)

    elif output_item[1].coo_sparse != None:
      # For SparseTensor op, Example: values_tensor_name: "CTCBeamSearchDecoder_1:1", indices_tensor_name: "CTCBeamSearchDecoder_1:0", dense_shape_tensor_name: "CTCBeamSearchDecoder_1:2"
      output_map1 = {"name": "", "dtype": 0, "shape": []}
      output_map2 = {"name": "", "dtype": 0, "shape": []}
      output_map3 = {"name": "", "dtype": 0, "shape": []}

      #values_tensor_name = output_item[1].coo_sparse.values_tensor_name
      #indices_tensor_name = output_item[1].coo_sparse.indices_tensor_name
      #dense_shape_tensor_name = output_item[1].coo_sparse.dense_shape_tensor_name

      values_op_name = "{}_{}".format(output_item[0], "values")
      indices_op_name = "{}_{}".format(output_item[0], "indices")
      shape_op_name = "{}_{}".format(output_item[0], "shape")
      output_map1["name"] = values_op_name
      output_map2["name"] = indices_op_name
      output_map3["name"] = shape_op_name

      # TODO: Add dtype and shape for sparse model
      model_graph_signature_dict["outputs"].append(output_map1)
      model_graph_signature_dict["outputs"].append(output_map2)
      model_graph_signature_dict["outputs"].append(output_map3)

  return model_graph_signature_dict
