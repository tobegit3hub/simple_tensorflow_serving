
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import traceback
import logging
import os
import time
import json
import numpy as np
from collections import namedtuple

from .abstract_inference_service import AbstractInferenceService


# Lazy init
NUMPY_DTYPE_MAP = {}


class PytorchOnnxInferenceService(AbstractInferenceService):
  """
  The service to load ONNX model and make inference with pytorch-caffe2 backend.
  """

  def __init__(self, model_name, model_base_path, verbose=False):
    """
    Initialize the service.
        
    Args:
      model_name: The name of the model.
      model_base_path: The file path of the model.
    Return:
      None
    """
    super(PytorchOnnxInferenceService, self).__init__()

    # Init onnx datatype map to numpy
    self.init_dtype_map()

    self.model_name = model_name
    self.model_base_path = model_base_path
    self.model_version_list = []

    self.model_version_dict = {}
    self.model_dict = {}
    self.executor_dict = {}
    self.model_graph_signature_dict = {}

    # This property is for index.html
    self.model_graph_signature = self.model_graph_signature_dict

    self.platform = "PyTorch_ONNX"
    self.verbose = verbose

    # Find available models
    model_path_list = []
    if os.path.isdir(self.model_base_path):
      for filename in os.path.listdir(self.model_base_path):
        if filename.endswith(".onnx"):
          path = os.path.join(self.model_base_path, filename)
          logging.info("Found onnx model: {}".format(path))
          model_path_list.append(path)
      if len(model_path_list) == 0:
        logging.error("No onnx model found in {}".format(self.model_base_path))
    elif os.path.isfile(self.model_base_path):
      logging.info("Found onnx model: {}".format(self.model_base_path))
      model_path_list.append(self.model_base_path)
    else:
      raise Exception("Invalid model_base_path: {}".format(self.model_base_path))

    # Load available models
    count = 1
    for model_path in model_path_list:
      try:
        version = str(count)
        model, executor, signature = self.load_model(model_path)
        logging.info("Load onnx model: {}, signature: {}".format(
          model_path, json.dumps(signature)))
        self.model_version_dict[version] = model_path
        self.model_dict[version] = model
        self.executor_dict[version] = executor
        self.model_graph_signature_dict[version] = signature
        self.model_version_list.append(version)
        count += 1
      except Exception as e:
        traceback.print_exc()

  def init_dtype_map(self):
    from onnx import TensorProto as tp
    global NUMPY_DTYPE_MAP
    NUMPY_DTYPE_MAP.update({
      tp.FLOAT: "float32",
      tp.UINT8: "uint8",
      tp.INT8: "int8",
      tp.INT32: "int32",
      tp.INT64: "int64",
      tp.DOUBLE: "float64",
      tp.UINT32: "uint32",
      tp.UINT64: "uint64"
    })

  def load_model(self, model_path):
    # TODO: Import as needed and only once
    import onnx
    import caffe2.python.onnx.backend as backend
    
    # Load model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    
    # Genetate signature
    signature = {"inputs": [], "outputs": []}
    for input in model.graph.input:
      if isinstance(input, str) or isinstance(input, unicode):
        # maybe old version onnx proto
        signature["inputs"].append({"name": input})
      else:
        info = input.type.tensor_type
        input_meta = {
          "name": input.name,
          "dtype": int(info.elem_type),
          "shape": [(d.dim_value if d.HasField("dim_value") else -1) 
                    for d in info.shape.dim]
        }
        signature["inputs"].append(input_meta)
    for output in model.graph.output:
      if isinstance(output, str) or isinstance(output, unicode):
        # maybe old version onnx proto
        signature["outputs"].append({"name": output})
      else:
        info = output.type.tensor_type
        output_meta = {
          "name": output.name,
          "dtype": int(info.elem_type),
          "shape": [(d.dim_value if d.HasField("dim_value") else -1) 
                    for d in info.shape.dim]
        }
        signature["outputs"].append(output_meta)

    # Build model executor
    executor = backend.prepare(model)
    return model, executor, signature

  def inference(self, json_data):
    # Get version
    model_version = str(json_data.get("model_version", ""))
    if model_version.strip() == "":
      model_version = self.model_version_list[-1]

    input_data = json_data.get("data", "")
    if str(model_version) not in self.model_version_dict or input_data == "":
      logging.error("No model version: {} to serve".format(model_version))
      return "Fail to request the model version: {} with data: {}".format(
          model_version, input_data)
    else:
      logging.debug("Inference with json data: {}".format(json_data))

    signature = self.model_graph_signature_dict[model_version]
    inputs_signature = signature["inputs"]
    inputs = []
    if isinstance(input_data, dict):
      for input_meta in inputs_signature:
        name = input_meta["name"]
        onnx_type = input_meta["dtype"]
        if name not in input_data:
          logging.error("Cannot find input name: {}".format(name))
        else:
          data_item = input_data[name]
          if not isinstance(data_item, np.ndarray):
            data_item = np.asarray(data_item)
          if onnx_type in NUMPY_DTYPE_MAP:
            numpy_type = NUMPY_DTYPE_MAP[onnx_type]
            if numpy_type != data_item.dtype:
              data_item = data_item.astype(numpy_type)
          inputs.append(data_item)
    else:
      raise Exception("Invalid json input data")        

    start_time = time.time()
    executor = self.executor_dict[model_version]
    outputs = executor.run(inputs)
    
    result = {}
    for idx, output_meta in enumerate(signature["outputs"]):
      name = output_meta["name"]
      result[name] = outputs[idx]
    logging.debug("Inference result: {}".format(result))
    return result

