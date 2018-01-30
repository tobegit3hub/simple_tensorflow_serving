import logging

import gen_python


def gen_tensorflow_sdk(tensorflow_inference_service, language):
  """
  Generate the TensorFlow SDK for programming languages.
  
  Args:
    tensorflow_inference_service: The tensorflow service object.
    language: The sdk in this programming language to generate.
    
  Return:
    None
  """

  print(tensorflow_inference_service)
  print(language)

  if language not in ["python"]:
    logging.error("Language: {} is not supported to gen sdk".format(language))
    return

  if language == "python":
    #import ipdb;ipdb.set_trace()
    gen_python.gen_tensorflow_python_sdk()
