
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
import json
import numpy as np
from collections import namedtuple

from .abstract_inference_service import AbstractInferenceService


class H2oInferenceService(AbstractInferenceService):
  """
  The H2O service to load H2O model and make inference.
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
    super(H2oInferenceService, self).__init__()

    self.model_name = model_name
    self.model_base_path = model_base_path
    self.model_version_list = [1]
    self.model_graph_signature = ""
    self.platform = "H2o"
    self.verbose = verbose

    import h2o

    logging.info("Try to initialize and connect the h2o server")
    h2o.init()

    logging.info("Try to load the h2o model")
    model = h2o.load_model(model_base_path)

    self.model = model
    # TODO: Update the signature with readable string
    self.model_graph_signature = "{}".format(self.model.full_parameters)

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

    # 1. Import libraries
    import h2o
    import pandas as pd

    # 2. Do inference
    request_ndarray_data = json_data["data"]["data"]

    if self.verbose:
      start_time = time.time()

    df = pd.read_json(json.dumps(request_ndarray_data), orient="index")

    #test = h2o.H2OFrame(df.values.tolist())
    test = h2o.H2OFrame(df)

    predictions = self.model.predict(test)
    predictions.show()

    #performance = self.model.model_performance(test)
    #performance.show()

    if self.verbose:
      logging.debug("Inference time: {} s".format(time.time() - start_time))

    result_df = predictions.as_data_frame()
    result_string = result_df.to_json(orient='index')

    # 3. Build return data
    result = json.loads(result_string)

    if self.verbose:
      logging.debug("Inference result: {}".format(result))
    return result
