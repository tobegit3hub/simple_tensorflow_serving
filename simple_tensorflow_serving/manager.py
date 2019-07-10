#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import json
import logging

import base64_util
from tensorflow_inference_service import TensorFlowInferenceService
from mxnet_inference_service import MxnetInferenceService
from onnx_inference_service import OnnxInferenceService
from pytorch_onnx_inference_service import PytorchOnnxInferenceService
from h2o_inference_service import H2oInferenceService
from scikitlearn_inference_service import ScikitlearnInferenceService
from xgboost_inference_service import XgboostInferenceService
from pmml_inference_service import PmmlInferenceService
from spark_inference_service import SparkInferenceService

logger = logging.getLogger('simple_tensorflow_serving')


class InferenceServiceManager:
  def __init__(self, args):
    self.args = args

    # Example: {"default": TensorFlowInferenceService}
    self.model_name_service_map = {}

  def init(self):

    if self.args.model_config_file != "":
      # Read from configuration file
      with open(self.args.model_config_file) as data_file:
        model_config_file_dict = json.load(data_file)
        # Example: [{u'platform': u'tensorflow', u'name': u'tensorflow_template_application', u'base_path': u'/Users/tobe/code/simple_tensorflow_serving/models/tensorflow_template_application_model/'}, {u'platform': u'tensorflow', u'name': u'deep_image_model', u'base_path': u'/Users/tobe/code/simple_tensorflow_serving/models/deep_image_model/'}]
        model_config_list = model_config_file_dict["model_config_list"]

        for model_config in model_config_list:
          # Example: {"name": "tensorflow_template_application", "base_path": "/", "platform": "tensorflow"}
          model_name = model_config["name"]
          model_base_path = model_config["base_path"]
          model_platform = model_config.get("platform", "tensorflow")
          custom_op_paths = model_config.get("custom_op_paths", "")
          session_config = model_config.get("session_config", {})

          if model_platform == "tensorflow":
            inference_service = TensorFlowInferenceService(
                model_name, model_base_path, custom_op_paths, session_config)
          elif model_platform == "mxnet":
            inference_service = MxnetInferenceService(model_name,
                                                      model_base_path)
          elif model_platform == "onnx":
            inference_service = OnnxInferenceService(model_name,
                                                     model_base_path)
          elif model_platform == "pytorch_onnx":
            inference_service = PytorchOnnxInferenceService(
                model_name, model_base_path)
          elif model_platform == "h2o":
            inference_service = H2oInferenceService(model_name,
                                                    model_base_path)
          elif model_platform == "scikitlearn":
            inference_service = ScikitlearnInferenceService(
                model_name, model_base_path)
          elif model_platform == "xgboost":
            inference_service = XgboostInferenceService(
                model_name, model_base_path)
          elif model_platform == "pmml":
            inference_service = PmmlInferenceService(model_name,
                                                     model_base_path)
          elif model_platform == "spark":
            inference_service = SparkInferenceService(model_name,
                                                      model_base_path)

          self.model_name_service_map[model_name] = inference_service
    else:

      # Read from command-line parameter
      if self.args.model_platform == "tensorflow":
        session_config = json.loads(self.args.session_config)
        inference_service = TensorFlowInferenceService(
            self.args.model_name, self.args.model_base_path,
            self.args.custom_op_paths, session_config)
      elif self.args.model_platform == "mxnet":
        inference_service = MxnetInferenceService(self.args.model_name,
                                                  self.args.model_base_path)
      elif self.args.model_platform == "h2o":
        inference_service = H2oInferenceService(self.args.model_name,
                                                self.args.model_base_path)
      elif self.args.model_platform == "onnx":
        inference_service = OnnxInferenceService(self.args.model_name,
                                                 self.args.model_base_path)
      elif self.args.model_platform == "pytorch_onnx":
        inference_service = PytorchOnnxInferenceService(
            self.args.model_name, self.args.model_base_path)
      elif self.args.model_platform == "scikitlearn":
        inference_service = ScikitlearnInferenceService(
            self.args.model_name, self.args.model_base_path)
      elif self.args.model_platform == "xgboost":
        inference_service = XgboostInferenceService(self.args.model_name,
                                                    self.args.model_base_path)
      elif self.args.model_platform == "pmml":
        inference_service = PmmlInferenceService(self.args.model_name,
                                                 self.args.model_base_path)
      elif self.args.model_platform == "spark":
        inference_service = SparkInferenceService(self.args.model_name,
                                                  self.args.model_base_path)

      self.model_name_service_map[self.args.model_name] = inference_service

    # Start thread to periodically reload models or not
    if self.args.reload_models == "True" or self.args.reload_models == "true":
      for model_name, inference_service in self.model_name_service_map.items():
        if inference_service.platform == "tensorflow":
          inference_service.dynamically_reload_models()

  def inference(self, model_name, json_data):
    """
    Args:
      save_file_dir: Path to save data.
  
    Return:
      json_data: The inference result or error message.
      status code: The HTTP response code.
    """

    inferenceService = self.model_name_service_map[model_name]

    if self.args.enable_b64_autoconvert:
      # Decode base64 string and modify request json data
      base64_util.replace_b64_in_dict(json_data)

    result = inferenceService.inference(json_data)
    return result
