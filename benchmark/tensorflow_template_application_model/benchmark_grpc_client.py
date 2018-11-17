#!/usr/bin/env python

import time
import numpy
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2


def main():
  host = "0.0.0.0"
  port = 8502
  model_name = "default"
  model_version = -1
  signature_name = ""
  request_timeout = 10.0

  # Generate inference data
  keys = numpy.asarray([[1]])
  keys_tensor_proto = tf.contrib.util.make_tensor_proto(keys, dtype=tf.int32)
  features = numpy.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
  features_tensor_proto = tf.contrib.util.make_tensor_proto(
      features, dtype=tf.float32)

  # Create gRPC client
  channel = implementations.insecure_channel(host, port)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  if model_version > 0:
    request.model_spec.version.value = model_version
  if signature_name != "":
    request.model_spec.signature_name = signature_name
  request.inputs["keys"].CopyFrom(keys_tensor_proto)
  request.inputs["features"].CopyFrom(features_tensor_proto)

  # Send request
  start_time = time.time()
  for i in range(100):
    result = stub.Predict(request, request_timeout)
  end_time = time.time()
  print("Cost time: {}".format(end_time - start_time))

  print(result)


if __name__ == "__main__":
  main()
