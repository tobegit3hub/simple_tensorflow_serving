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

  def __init__(self, model_name, model_base_path, custom_op_paths="", verbose=False):
    """
    Initialize the service.
        
    Args:
      model_base_path: The file path of the model.
      model_name: The name of the model.
      model_version: The version of the model.
    Return:
    """

    super(MxnetInferenceService, self).__init__()

    self.model_name = model_name
    self.model_base_path = model_base_path
    self.model_version_list = [1]
    self.model_graph_signature = ""
    self.platform = "MXNet"

    # TODO: Import as needed and only once
    import mxnet as mx

    # TODO: Select the available version
    epoch_number = 1

    # Load model
    sym, arg_params, aux_params = mx.model.load_checkpoint(self.model_base_path, epoch_number)
    self.mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    self.mod.bind(for_training=False, data_shapes=[('data', (1L,2L))])
    self.mod.set_params(arg_params, aux_params, allow_missing=True)
    self.model_graph_signature = self.mod.symbol.tojson()

    self.verbose = verbose


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

    # 1. Build inference data
    Batch = namedtuple('Batch', ['data'])
    # batch = Batch([mx.nd.array([[7.0, 2.0]])])
    request_ndarray_data = json_data["data"]["data"]
    request_mxnet_ndarray_data = [mx.nd.array(request_ndarray_data)]
    batch_data = Batch(request_mxnet_ndarray_data)

    # 2. Do inference
    mod.forward(batch_data)
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
