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
