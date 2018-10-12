# Quick Start

Start the server with the TensorFlow [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model).

```
simple_tensorflow_serving --model_base_path="./models/tensorflow_template_application_model"
```

Check out the dashboard in [http://127.0.0.1:8500](http://127.0.0.1:8500) in web browser.
 
![dashboard](./images/dashboard.png)

Generate Python client and access the model with test data without coding.

```
curl http://localhost:8500/v1/models/default/gen_client?language=python > client.py
```

```
python ./client.py
```