#!/bin/bash

set -x
set -e

cd ../simple_tensorflow_serving/
./server.py --model_base_path="../models/tensorflow_template_application_model/"  > log 2>&1
