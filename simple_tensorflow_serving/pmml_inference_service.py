from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import numpy as np
import pickle

from .abstract_inference_service import AbstractInferenceService


class PmmlInferenceService(AbstractInferenceService):
  """
  The service to load PMML model and make inference.
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

    super(PmmlInferenceService, self).__init__()

    self.model_name = model_name
    self.model_base_path = model_base_path
    self.model_version_list = [1]
    self.model_graph_signature = ""
    self.platform = "PMML"

    # Load model
    from openscoring import Openscoring
    openscoring_server_endpoint = "localhost:8080"
    kwargs = {"auth" : ("admin", "adminadmin")}
    self.openscoring = Openscoring("http://{}/openscoring".format(openscoring_server_endpoint))
    self.openscoring.deployFile("PmmlModel", self.model_base_path, **kwargs)

    self.model_graph_signature = "No signature for PMML models"
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

    # 1. Build inference data
    # Example: arguments = {"Sepal_Length" : 5.1, "Sepal_Width" : 3.5, "Petal_Length" : 1.4, "Petal_Width" : 0.2}
    request_json_data =  json_data["data"]

    # 2. Do inference
    if self.verbose:
      start_time = time.time()

    # Example: {u'Probability_setosa': 1.0, u'Probability_versicolor': 0.0, u'Node_Id': u'2', u'Species': u'setosa', u'Probability_virginica': 0.0}
    predict_result = self.openscoring.evaluate("PmmlModel", request_json_data)

    if self.verbose:
      logging.debug("Inference time: {} s".format(time.time() - start_time))

    # 3. Build return data
    result = {
        "result": predict_result,
    }

    if self.verbose:
      logging.debug("Inference result: {}".format(result))

    return result
