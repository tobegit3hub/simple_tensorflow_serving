#!/bin/bash

set -x
set -e

docker run -it -v /:/host -p 8500:8500  docker02:35000/operator-repository/tensorflow-serving-cpu  bash
/tensorflow_model_server --port 8501 --rest_api_port 8500 --model_base_path="/host/Users/tobe/code/simple_tensorflow_serving/models/tensorflow_template_application_model"
