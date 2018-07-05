#!/bin/bash

set -x
set -e

cd ../simple_tensorflow_serving/
uwsgi --http 0.0.0.0:8500 --threads 4 -w wsgi --pyargv "--model_base_path ../models/tensorflow_template_application_model" > log 2>&1
