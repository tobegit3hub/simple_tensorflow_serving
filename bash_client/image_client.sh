#!/bin/bash

curl -X POST -F 'image=@../examples/example.jpg' -F "model_version=1" 127.0.0.1:8500
