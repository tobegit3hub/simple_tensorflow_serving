# Simple TensorFlow Serving

![](./images/simple_tensorflow_serving_introduction.jpeg)

## Introduction

Simple TensorFlow Serving is the generic and easy-to-use serving service for machine learning models.

It is the bridge for TensorFlow model and bring machine learning to any programming language, such as [Python](./python_client/), [C++](./cpp_client/), [Java](./java_client/), [Scala](./scala_client/), [Go](./go_client/), [Ruby](./ruby_client), [JavaScript](./javascript_client/), [PHP](./php_client/), [Erlang](./erlang_client/), [Lua](./lua_client/), [Rust](./rust_client/), [Swift](./swift_client/), [Lisp](./lisp_client/), [Haskell](./haskell_client/) and so on.

* [x] Support arbitrary TensorFlow models
* [x] Support the general RESTful/HTTP APIs
* [x] Support `curl` and other command-line tools
* [x] Support clients in any programming language
* [x] Support statistical metrics for verbose requests
* [ ] Support loading multiple TF models dynamically

## Installation

Install the server with `pip`.

```shell
pip install simple-tensorflow-serving
```

Or install with `bazel`.

```shell
bazel build simple_tensorflow_serving:server
```

Or install from source code.

```shell
python ./setup.py install
```

## Usage

You can export [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model) and setup the server easily.

```shell
simple_tensorflow_serving --port=8500 --model_base_path="./examples/tensorflow_template_application_model"
```

Then request TensorFlow serving with `curl`.

```shell
curl -H "Content-Type: application/json" -X POST -d '{"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}' http://127.0.0.1:8500
```

Here is the example client in [Python](./python_client/).

```python
endpoint = "http://127.0.0.1:8500"
payload = {"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}

result = requests.post(endpoint, json=payload)
```

Here is the example client in [C++](./cpp_client/).

Here is the example client in [Java](./java_client/).

Here is the example client in [Scala](./scala_client/).

Here is the example client in [Go](./go_client/).

```go
endpoint := "http://127.0.0.1:8500"
dataByte := []byte(`{"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}`)
var dataInterface map[string]interface{}
json.Unmarshal(dataByte, &dataInterface)
dataJson, _ := json.Marshal(dataInterface)

resp, err := http.Post(endpoint, "application/json", bytes.NewBuffer(dataJson))
```

Here is the example client in [Ruby](./ruby_client/).

```ruby
endpoint = "http://127.0.0.1:8500"
uri = URI.parse(endpoint)
header = {"Content-Type" => "application/json"}
input_data = {"keys"=> [[11.0], [2.0]], "features"=> [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
http = Net::HTTP.new(uri.host, uri.port)
request = Net::HTTP::Post.new(uri.request_uri, header)
request.body = input_data.to_json

response = http.request(request)
```

Here is the example client in [JavaScript](./javascript_client/).

```javascript
var options = {
    uri: "http://127.0.0.1:8500",
    method: "POST",
    json: {"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
};

request(options, function (error, response, body) {});
```

Here is the example client in [PHP](./php_client/).

```php
$endpoint = "127.0.0.1:8500";
$postData = array(
    "keys" => [[11.0], [2.0]],
    "features" => [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
);
$ch = curl_init($endpoint);
curl_setopt_array($ch, array(
    CURLOPT_POST => TRUE,
    CURLOPT_RETURNTRANSFER => TRUE,
    CURLOPT_HTTPHEADER => array(
        "Content-Type: application/json"
    ),
    CURLOPT_POSTFIELDS => json_encode($postData)
));

$response = curl_exec($ch);

```

Here is the example client in [Erlang](./erlang_client/).

```erlang
ssl:start(),
application:start(inets),

httpc:request(post,
    {"http://127.0.0.1:8500", [],
    "application/json",
    "{\"keys\": [[11.0], [2.0]], \"features\": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}"
    }, [], []).
```

Here is the example client in [Lua](./lua_client/).

```lua
local endpoint = "http://127.0.0.1:8500"
keys_array = {}
keys_array[1] = {1.0}
keys_array[2] = {2.0}
features_array = {}
features_array[1] = {1, 1, 1, 1, 1, 1, 1, 1, 1}
features_array[2] = {1, 1, 1, 1, 1, 1, 1, 1, 1}
local input_data = {
    ["keys"] = keys_array,
    ["features"] = features_array,
}
request_body = json:encode (input_data)
local response_body = {}

local res, code, response_headers = http.request{
    url = endpoint,
    method = "POST", 
    headers = 
      {
          ["Content-Type"] = "application/json";
          ["Content-Length"] = #request_body;
      },
      source = ltn12.source.string(request_body),
      sink = ltn12.sink.table(response_body),
}
```

Here is the example client in [Rust](./swift_client/).

Here is the example client in [Swift](./swift_client/).

Here is the example client in [Lisp](./swift_client/).

Here is the example client in [Haskell](./swift_client/).

Use your favorite HTTP clients like `Postman`.

![](./images/simple_tensorflow_serving_client.png)

## How It Works

1. `simple_tensorflow_serving` starts the HTTP server with `flask` application.
2. Load the TensorFlow models with `tf.saved_model.loader` Python APIs.
3. Construct the feed_dict data from the JSON body of the request
4. Use the TensorFlow Python API to `sess.run()` with feed_dict data
5. For multiple versions supported, it has independent thread to load models

![](./images/architecture.jpeg)

## Contribution

Check out the C++ implementation of TensorFlow Serving in [tensorflow/serving](https://github.com/tensorflow/serving).

Feel free to open an issue or send pull request for this project.
