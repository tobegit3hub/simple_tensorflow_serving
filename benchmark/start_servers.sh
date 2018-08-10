#!/bin/bash

set -x
set -e

# Simple_tensorflow_serving with Flask
docker run -it -p 8500:8500 -m=10g --cpus=10 -v /Users/tobe/code/simple_tensorflow_serving:/host simple-tensorflow-serving-cpu bash

simple_tensorflow_serving --model_base_path="/host/benchmark/simplest_model/model/"
simple_tensorflow_serving --model_base_path="/host/benchmark/tensorflow_template_application_model/tensorflow_template_application_model/"
simple_tensorflow_serving --model_base_path="/host/benchmark/inception_v4/inceptionv4_224_224_model/"
simple_tensorflow_serving --model_base_path="/host/benchmark/vgg_16/vgg16_224_224_model/"

# Simple_tensorflow_serving with UWSGI
docker run -it -p 8501:8501 -m=10g --cpus=10 -v /Users/tobe/code/simple_tensorflow_serving:/host simple-tensorflow-serving-cpu bash

cd /host/
uwsgi --http 0.0.0.0:8501 --threads 10 -w wsgi --pyargv "--model_base_path /host/benchmark/simplest_model/model/"
uwsgi --http 0.0.0.0:8501 --threads 10 -w wsgi --pyargv "--model_base_path /host/benchmark/tensorflow_template_application_model/tensorflow_template_application_model/"
uwsgi --http 0.0.0.0:8501 --threads 10 -w wsgi --pyargv "--model_base_path /host/benchmark/inception_v4/inceptionv4_224_224_model/"
uwsgi --http 0.0.0.0:8501 --threads 10 -w wsgi --pyargv "--model_base_path /host/benchmark/vgg_16/vgg16_224_224_model/"


# TensorFlow Serving with RESTful/gRPC
docker run -it -m=10g --cpus=10 -p 8502:8502 -p 8503:8503 -v /Users/tobe/code/simple_tensorflow_serving:/host tensorflow-serving-cpu bash

/tensorflow_model_server --port=8502 --rest_api_port=8503 --model_base_path="/host/benchmark/simplest_model/model/"
/tensorflow_model_server --port=8502 --rest_api_port=8503 --model_base_path="/host/benchmark/tensorflow_template_application_model/tensorflow_template_application_model/"
/tensorflow_model_server --port=8502 --rest_api_port=8503 --model_base_path="/host/benchmark/inception_v4/inceptionv4_224_224_model/"
/tensorflow_model_server --port=8502 --rest_api_port=8503 --model_base_path="/host/benchmark/vgg_16/vgg16_224_224_model/"

# Use official TensorFlow Serving GPU
#export CUDA_SO="-v /usr/cuda_files:/usr/cuda_files"
#export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
#$CUDA_SO $DEVICES
#export LD_LIBRARY_PATH=/usr/cuda_files:$LD_LIBRARY_PATH

