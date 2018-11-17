# API

## RESTful API

The most import API is inference for the loaded models.

```
Endpoint: /
Method: POST
JSON: {
        "model_name": "default",
        "model_version": 1,
        "data": {
            "keys": [[11.0], [2.0]],
            "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1]]
        }
      }
Response: {
            "keys": [[1], [1]]
          }
```


## Python Example

You can easily choose the specified model and version for inference.

```python
import requests

endpoint = "http://127.0.0.1:8500"
input_data = {
  "model_name": "default",
  "model_version": 1,
  "data": {
      "keys": [[11.0], [2.0]],
      "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1]]
  }
}
result = requests.post(endpoint, json=input_data)
```
