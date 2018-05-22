#!/usr/bin/env python

import tensorflow as tf


def main():
  test_file_path = "../models/tensorflow_template_application_model/1"
  is_saved_model = check_saved_model(test_file_path)
  if is_saved_model:
    print("{} is the saved model".format(test_file_path))
  else:
    print("{} is NOT the saved model".format(test_file_path))


def check_saved_model(model_file_path):
  try:
    session = tf.Session(graph=tf.Graph())
    meta_graph = tf.saved_model.loader.load(
        session, [tf.saved_model.tag_constants.SERVING], model_file_path)
    return True
  except IOError as ioe:
    print("Catch exception: ".foramt(ioe))
    return False


if __name__ == "__main__":
  main()
