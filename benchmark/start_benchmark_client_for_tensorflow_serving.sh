#!/bin/bash

ab -n 10000 -c 1 -T "application/json" -p ./tensorflow_template_application_data_for_tensorflow_serving.json http://127.0.0.1:8500/
