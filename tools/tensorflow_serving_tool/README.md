# TensorFlow Serving Tool

## Start TensorFlow Serving

```
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_base_path=/host/Users/tobe/code/simple_tensorflow_serving/models/tensorflow_template_application_model
```

## Use gRPC Client

```
./python_grpc_client.py --host 172.27.128.107 --port 8846 --model_name default --model_version 1
```

## Use HTTP Client

```
./python_client.py
```
