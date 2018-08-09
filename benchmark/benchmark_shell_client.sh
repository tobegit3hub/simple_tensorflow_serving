#!/bin/bash

set -x
set -e

# Simplest Model
curl -H "Content-Type: application/json" -X POST -d '{"data": {"keys": [[1]]}}' http://127.0.0.1:8500
ab -n 10000 -c 1 -T "application/json" -p ./data.json http://127.0.0.1:8500/

ab -n 10000 -c 1 -T "application/json" -p ./tensorflow_template_application_data.json http://127.0.0.1:8500/
