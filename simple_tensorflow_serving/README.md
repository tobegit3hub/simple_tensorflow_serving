# Server

## Introduction

Start the server with installed binary.

```
imple_tensorflow_serving --port=8500 --model_base_path="./models/tensorflow_template_application_model"
``

Or start with Python script.

```
imple_tensorflow_serving --port=8500 --model_base_path="./models/tensorflow_template_application_model"
``

Or start with [gunicorn](http://gunicorn.org/).

```
gunicorn --bind 0.0.0.0:8500 wsgi
```
