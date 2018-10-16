# API




Adding or removing model versions will be detected automatically and re-load latest files in memory. You can easily choose the specified model and version for inference.

```
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
