from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
import subprocess
import requests

from .abstract_inference_service import AbstractInferenceService
from . import filesystem_util
from . import preprocess_util

logger = logging.getLogger("simple_tensorflow_serving")


class PmmlInferenceService(AbstractInferenceService):
  """
  The service to load PMML model and make inference.
  """

  def __init__(self, model_name, model_base_path):
    """
    Initialize the service.
        
    Args:
      model_name: The name of the model.
      model_base_path: The file path of the model.
    Return:
      None
    """

    super(PmmlInferenceService, self).__init__()

    # Start the pmml server
    if os.path.isfile("/tmp/openscoring-server-executable-1.4-SNAPSHOT.jar"):
      logging.info(
          "Run to run command 'java -jar /tmp/openscoring-server-executable-1.4-SNAPSHOT.jar'"
      )
      subprocess.Popen([
          "java", "-jar", "/tmp/openscoring-server-executable-1.4-SNAPSHOT.jar"
      ])

      logging.info("Sleep 10s to wait for pmml server")
      time.sleep(10)

    local_model_base_path = filesystem_util.download_hdfs_moels(
        model_base_path)

    self.model_name = model_name
    self.model_base_path = local_model_base_path
    self.model_version_list = [1]
    self.model_graph_signature = ""
    self.platform = "PMML"

    self.preprocess_function, self.postprocess_function = preprocess_util.get_preprocess_postprocess_function_from_model_path(
        self.model_base_path)

    # Load model
    """
    from openscoring import Openscoring
    # TODO: Support to config the port of Openscoring server
    openscoring_server_endpoint = "localhost:8080"
    kwargs = {"auth": ("admin", "adminadmin")}
    self.openscoring = Openscoring(
        "http://{}/openscoring".format(openscoring_server_endpoint))
    self.openscoring.deployFile(self.model_name, self.model_base_path,
                                **kwargs)
    """

    endpoint = "http://localhost:8080/openscoring/model/{}".format(
        self.model_name)

    with open(self.model_base_path, "rb") as f:
      kwargs = {
          'headers': {
              'content-type': 'application/xml'
          },
          'json': None,
          'data': f,
          'auth': ('admin', 'adminadmin')
      }
      result = requests.put(endpoint, **kwargs)
      logging.info("Deploy the model to Openscoring: {}".format(result.text))

    self.model_graph_signature = "No signature for PMML models"

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
    request_json_data = json_data["data"]

    if json_data.get("preprocess", "false") != "false":
      if self.preprocess_function != None:
        request_json_data = self.preprocess_function(request_json_data)
        logger.debug(
            "Preprocess to generate data: {}".format(request_json_data))
      else:
        logger.warning("No preprocess function in model")

    # 2. Do inference
    start_time = time.time()
    # Example: {'Probability_setosa': 1.0, 'Probability_versicolor': 0.0, 'Node_Id': '2', 'Species': 'setosa', 'Probability_virginica': 0.0}
    #predict_result = self.openscoring.evaluate(self.model_name, request_json_data)

    endpoint = "http://localhost:8080/openscoring/model/{}".format(
        self.model_name)
    input_data = {"id": None, "arguments": request_json_data}
    result = requests.post(endpoint, json=input_data)
    predict_result = result.json()

    logger.debug("Inference time: {} s".format(time.time() - start_time))
    logger.debug("Inference result: {}".format(predict_result))

    if json_data.get("postprocess", "false") != "false":
      if self.postprocess_function != None:
        predict_result = self.postprocess_function(predict_result)
        logger.debug("Postprocess to generate data: {}".format(predict_result))
      else:
        logger.warning("No postprocess function in model")

    return predict_result
