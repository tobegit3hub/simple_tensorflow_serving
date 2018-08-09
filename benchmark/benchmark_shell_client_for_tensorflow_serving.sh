#!/bin/bash

set -x
set -e

# Simplest Model

curl -d '{"instances": [{"keys": 1}]}' -X POST http://127.0.0.1:8503/v1/models/default/versions/1:predict
ab -n 10000 -c 1 -T "application/json" -p ./data_for_tensorflow_serving.json http://127.0.0.1:8503/
