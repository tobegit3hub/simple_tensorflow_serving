#!/usr/bin/env python

import tensorflow as tf

import logging

logging.basicConfig(level=logging.DEBUG)


def check_saved_model(model_file_path):
  logging.info("Try to load the model in: {}".format(model_file_path))

  try:
    session = tf.Session(graph=tf.Graph())
    meta_graph = tf.saved_model.loader.load(
        session, [tf.saved_model.tag_constants.SERVING], model_file_path)
    logging.info("Succeed to load model in: {}".format(model_file_path))
  except IOError as ioe:
    logging.info("Fail to load model and catch exception: {}".format(ioe))


def main():
  model_file_path = "../models/tensorflow_template_application_model/1"
  check_saved_model(model_file_path)

  model_file_path = "../models/tensorflow_template_application_model/100"
  check_saved_model(model_file_path)


if __name__ == "__main__":
  main()
