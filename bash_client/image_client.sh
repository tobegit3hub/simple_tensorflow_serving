#!/bin/bash

#curl -X POST -F 'image=@../images/mew.jpg' -F "model_version=1" 127.0.0.1:8500

#curl -X POST -F 'image=@../images/mew.jpg' -F "model_version=1" -F "shape=1,32,32,3"  127.0.0.1:8500

curl -X POST -F 'image=@./1.jpg' -F "model_version=1" -F "shape=1,784"  127.0.0.1:8500
