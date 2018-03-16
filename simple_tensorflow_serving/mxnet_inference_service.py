import logging
import os
import time
import json
import numpy as np
from collections import namedtuple

from abstract_inference_service import AbstractInferenceService


class MxnetInferenceService(AbstractInferenceService):
  """
  The MXNet service to load MXNet checkpoint and make inference.
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
    self.mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    self.has_signature_file = False
    self.signature_input_names = []
    self.signature_output_names = []

    # Load inputs from signature file
    signature_file_path = self.model_base_path + "-signature.json"
    if os.path.exists(signature_file_path) and os.path.isfile(signature_file_path):
      self.has_signature_file = True
      data_shapes = []

      with open(signature_file_path) as signature_file:
        signature_dict = json.load(signature_file)
        inputs = signature_dict["inputs"]
        outputs = signature_dict["outputs"]

        for input in inputs:
          input_data_name = input["data_name"]
          input_data_shape = input["data_shape"]

          self.signature_input_names.append(input_data_name)
          data_shapes.append((input_data_name, tuple(input_data_shape)))

        for output in outputs:
          output_data_name = output["data_name"]
          ouput_data_shape = output["data_shape"]

          self.signature_output_names.append(output_data_name)

    else:
      data_shapes=[('data', (1, 2))]

    self.mod.bind(for_training=False, data_shapes=data_shapes)
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
    from mxnet.io import DataBatch

    # 1. Build inference data

    # TODO: Should use DataBatch or not
    # from mxnet.io import DataBatch
    # request_ndarray_data = json_data["data"]["data"]
    # request_mxnet_ndarray_data = [mx.nd.array(request_ndarray_data)]
    # batch_data = DataBatch(request_mxnet_ndarray_data)

    if self.has_signature_file:
      request_mxnet_ndarray_data = []
      batch_tuple_list = []

      for input_name in self.signature_input_names:
        # Example: [[7.0, 2.0]]
        request_ndarray_data = json_data["data"][input_name]
        batch_tuple_list.append(input_name)
        request_mxnet_ndarray_data.append(mx.nd.array(request_ndarray_data))

      # namedtuple('Batch', ['data'])
      Batch = namedtuple('Batch', batch_tuple_list)
      batch_data = Batch(request_mxnet_ndarray_data)
    else:
      Batch = namedtuple('Batch', ['data'])
      request_ndarray_data = json_data["data"]["data"]
      request_mxnet_ndarray_data = [mx.nd.array(request_ndarray_data)]
      batch_data = Batch(request_mxnet_ndarray_data)


    # 2. Do inference
    if self.verbose:
      start_time = time.time()

    self.mod.forward(batch_data)

    if self.verbose:
      logging.debug("Inference time: {} s".format(time.time() - start_time))

    model_outputs = self.mod.get_outputs()
    prob = self.mod.get_outputs()[0].asnumpy()
    print(prob)

    # 3. Build return data
    result = {}
    for i, model_output in enumerate(model_outputs):
      result[self.signature_output_names[i]] = model_output.asnumpy()

    if self.verbose:
      logging.debug("Inference result: {}".format(result))
    return result
