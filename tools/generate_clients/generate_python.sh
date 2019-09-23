#!/bin/bash

curl http://localhost:8500/v1/models/default/gen_client?language=python > client.py

chmod +x ./client.py
