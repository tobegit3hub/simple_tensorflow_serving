import logging
import os
import time
import numpy as np
from collections import namedtuple

from abstract_inference_service import AbstractInferenceService


class MxnetInferenceService(AbstractInferenceService):
  """
  The MXNet service to load MXNet checkpoint and make inference.
  """

  def __init__(self, model_base_path, custom_op_paths="", verbose=False):
    """
    Initialize the service.
        
    Args:
      model_base_path: The file path of the model.
      model_name: The name of the model.
      model_version: The version of the model.
    Return:
    """

    import mxnet as mx

    self.model_base_path = model_base_path

    self.version_session_map = {}

    self.model_name_version_session_map = {}

    self.model_graph_signature = None

    self.model_name_graph_signature_map = {}

    self.verbose = verbose
    self.should_stop_all_threads = False


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

    import mxnet as mx
    # model_version = 1

    Batch = namedtuple('Batch', ['data'])

    # TODO: Load from model_base_path
    sym, arg_params, aux_params = mx.model.load_checkpoint('mx_mlp', 50)

    # TODO: Not work
    #sym = mx.symbol.load('/Users/tobe/code/simple_tensorflow_serving/models/mxnet_mlp/mx_mlp-symbol.json')

    #mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())

    #mod.bind(for_training=False, data_shapes=[('data', (1L,2L))],
    #         label_shapes=mod._label_shapes)
    mod.bind(for_training=False, data_shapes=[('data', (1L,2L))])

    mod.set_params(arg_params, aux_params, allow_missing=True)

    # 1. Build inference data
    # batch = Batch([mx.nd.array([[7.0, 2.0]])])
    batch = Batch([mx.nd.array( json_data["data"]["data"])])

    # 2. Do inference
    mod.forward(batch)
    model_outputs = mod.get_outputs()
    prob = mod.get_outputs()[0].asnumpy()
    print(prob)

    # 3. Build return data
    result = {}
    for i, model_output in enumerate(model_outputs):
      result[str(i)] = model_output.asnumpy()

    if self.verbose:
      logging.debug("Inference result: {}".format(result))
    return result
