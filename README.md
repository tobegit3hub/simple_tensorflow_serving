# Simple TensorFlow Serving

## Introduction

The simpler and easy-to-use serving service for general TensorFlow models.

* [x] Support TensorFlow SavedModel
* [x] Support the RESTful/HTTP APIs
* [x] Support `curl` and command-line tools
* [x] Support clients in any programing language

## Installation

Install the server with `pip`.

```
pip install simple-tensorflow-serving
```

Or install with `bazel`.

```
bazel build simple_tensorflow_serving:server
```

Or install from source code.

```
python ./setup.py install
```

## Usage

You can export the [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model) and setup the server easily.

```
simple_tensorflow_serving --port=8500 --model_base_path="./examples/tensorflow_template_application_model"
```

Then request with your favorite HTTP clients.

```
curl -H "Content-Type: application/json" -X POST -d '{"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}' http://127.0.0.1:8500
```

![](./images/simple_tensorflow_serving_client.png)

## Contribution

Check out the C++ implementation of TensorFlow Serving in [tensorflow/serving](https://github.com/tensorflow/serving).

Feel free to open an issue or send pull request for this project.
