from abc import ABCMeta, abstractmethod


class AbstractInferenceService(object):
  """
  The abstract class for inference service which should implement the method.
  """

  __metaclass__ = ABCMeta

  @abstractmethod
  def inference(self, input_data):
    """
    Args:
      input_data: The JSON serialized object with key and array data.
    Return:
      The JSON serialized object with key and array data.
    """
    pass
