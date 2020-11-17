#!/usr/bin/env python

import tensorflow as tf


def check_saved_model(model_file_path):
  print("Try to check the model in: {}".format(model_file_path))

  try:
    session = tf.Session(graph=tf.Graph())
    meta_graph = tf.saved_model.loader.load(
        session, [tf.saved_model.tag_constants.SERVING], model_file_path)
    print("It is valid model in: {}".format(model_file_path))
  except IOError as e:
    print("It is not valid model and catch exception: {}".format(e))


if __name__ == "__main__":
  model_file_path = "../models/tensorflow_template_application_model/1"
  check_saved_model(model_file_path)
