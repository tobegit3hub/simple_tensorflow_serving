#!/bin/bash

set -x
set -e

# 1 client
sleep 3
echo "Benchmark flask"
ab -n 10000 -c 1 -T "application/json" -p ./data.json http://127.0.0.1:8500/

sleep 3
echo "Benchmark uwsgi"
ab -n 10000 -c 1 -T "application/json" -p ./data.json http://127.0.0.1:8501/

sleep 3
echo "Benchmark tensorflow serving restful"
ab -n 10000 -c 1 -T "application/json" -p ./data_for_tensorflow_serving.json http://127.0.0.1:8503/v1/models/default/versions/1:predict

# 10 client
sleep 3
echo "Benchmark flask"
ab -n 10000 -c 10 -T "application/json" -p ./data.json http://127.0.0.1:8500/

sleep 3
echo "Benchmark uwsgi"
ab -n 10000 -c 10 -T "application/json" -p ./data.json http://127.0.0.1:8501/

sleep 3
echo "Benchmark tensorflow serving restful"
ab -n 10000 -c 10 -T "application/json" -p ./data_for_tensorflow_serving.json http://127.0.0.1:8503/v1/models/default/versions/1:predict

# 50 client
sleep 3
echo "Benchmark flask"
ab -n 10000 -c 50 -T "application/json" -p ./data.json http://127.0.0.1:8500/

sleep 3
echo "Benchmark uwsgi"
ab -n 10000 -c 50 -T "application/json" -p ./data.json http://127.0.0.1:8501/

sleep 3
echo "Benchmark tensorflow serving restful"
ab -n 10000 -c 50 -T "application/json" -p ./data_for_tensorflow_serving.json http://127.0.0.1:8503/v1/models/default/versions/1:predict

# 100 client
sleep 3
echo "Benchmark flask"
ab -n 10000 -c 100 -T "application/json" -p ./data.json http://127.0.0.1:8500/

sleep 3
echo "Benchmark uwsgi"
ab -n 10000 -c 100 -T "application/json" -p ./data.json http://127.0.0.1:8501/

sleep 3
echo "Benchmark tensorflow serving restful"
ab -n 10000 -c 100 -T "application/json" -p ./data_for_tensorflow_serving.json http://127.0.0.1:8503/v1/models/default/versions/1:predict


