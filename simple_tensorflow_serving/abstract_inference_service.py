from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class AbstractInferenceService(object):
  """
  The abstract class for inference service which should implement the method.
  """

  __metaclass__ = ABCMeta

  def __init__(self):
    self.model_name = None
    self.model_base_path = ""
    self.model_version_list = []
    self.model_graph_signature = None
    self.model_graph_signature_dict = {}
    self.platform = ""

  @abstractmethod
  def inference(self, json_data):
    """
    Args:
      json_data: The JSON serialized object with key and array data.
    Return:
      The JSON serialized object with key and array data.
    """
    pass

  def get_detail(self):
    detail = {}
    detail["model_name"] = self.model_name
    detail["model_base_path"] = self.model_base_path
    detail["model_version_list"] = self.model_version_list
    detail["model_signature"] = self.model_graph_signature_dict
    detail["platform"] = self.platform
    return detail