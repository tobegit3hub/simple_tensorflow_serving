# Models

## TensorFlow

For TensorFlow models, you can load with commands and configuration like these.

```
simple_tensorflow_serving --model_base_path="./models/tensorflow_template_application_model" --model_platform="tensorflow"
```

```python
endpoint = "http://127.0.0.1:8500"
input_data = {
  "model_name": "default",
  "model_version": 1,
  "data": {
      "data": [[12.0, 2.0]]
  }
}
result = requests.post(endpoint, json=input_data)
print(result.text)
```

## MXNET

For MXNet models, you can load with commands and configuration like these.

```
simple_tensorflow_serving --model_base_path="./models/mxnet_mlp/mx_mlp" --model_platform="mxnet"
```

```python
endpoint = "http://127.0.0.1:8500"
input_data = {
  "model_name": "default",
  "model_version": 1,
  "data": {
      "data": [[12.0, 2.0]]
  }
}
result = requests.post(endpoint, json=input_data)
print(result.text)
```

## ONNX

For ONNX models, you can load with commands and configuration like these.

```
simple_tensorflow_serving --model_base_path="./models/onnx_mnist_model/onnx_model.proto" --model_platform="onnx"
```

```python
endpoint = "http://127.0.0.1:8500"
input_data = {
  "model_name": "default",
  "model_version": 1,
  "data": {
      "data": [[...]]
  }
}
result = requests.post(endpoint, json=input_data)
print(result.text)
```

## Scikit-learn

For Scikit-learn models, you can load with commands and configuration like these.

```
simple_tensorflow_serving --model_base_path="./models/scikitlearn_iris/model.joblib" --model_platform="scikitlearn"

simple_tensorflow_serving --model_base_path="./models/scikitlearn_iris/model.pkl" --model_platform="scikitlearn"
```

```python
endpoint = "http://127.0.0.1:8500"
input_data = {
  "model_name": "default",
  "model_version": 1,
  "data": {
      "data": [[...]]
  }
}
result = requests.post(endpoint, json=input_data)
print(result.text)
```

## XGBoost

For XGBoost models, you can load with commands and configuration like these.

```
simple_tensorflow_serving --model_base_path="./models/xgboost_iris/model.bst" --model_platform="xgboost"

simple_tensorflow_serving --model_base_path="./models/xgboost_iris/model.joblib" --model_platform="xgboost"

simple_tensorflow_serving --model_base_path="./models/xgboost_iris/model.pkl" --model_platform="xgboost"
```

```python
endpoint = "http://127.0.0.1:8500"
input_data = {
  "model_name": "default",
  "model_version": 1,
  "data": {
      "data": [[...]]
  }
}
result = requests.post(endpoint, json=input_data)
print(result.text)
```


## PMML

For PMML models, you can load with commands and configuration like these. This relies on [Openscoring](https://github.com/openscoring/openscoring) and [Openscoring-Python](https://github.com/openscoring/openscoring-python) to load the models.

```
java -jar ./third_party/openscoring/openscoring-server-executable-1.4-SNAPSHOT.jar

simple_tensorflow_serving --model_base_path="./models/pmml_iris/DecisionTreeIris.pmml" --model_platform="pmml"
```

```python
endpoint = "http://127.0.0.1:8500"
input_data = {
  "model_name": "default",
  "model_version": 1,
  "data": {
      "data": [[...]]
  }
}
result = requests.post(endpoint, json=input_data)
print(result.text)
```


## H2o

For H2o models, you can load with commands and configuration like these.

```
# Start H2o server with "java -jar h2o.jar"

simple_tensorflow_serving --model_base_path="./models/h2o_prostate_model/GLM_model_python_1525255083960_17" --model_platform="h2o"
```

```python
endpoint = "http://127.0.0.1:8500"
input_data = {
  "model_name": "default",
  "model_version": 1,
  "data": {
      "data": [[...]]
  }
}
result = requests.post(endpoint, json=input_data)
print(result.text)
```
