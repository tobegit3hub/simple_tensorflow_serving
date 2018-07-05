# Server

## Introduction

Start the server with installed binary.

```
simple_tensorflow_serving --port=8500 --model_base_path="./models/tensorflow_template_application_model"
``

Or start with Python script.

```
simple_tensorflow_serving --port=8500 --model_base_path="./models/tensorflow_template_application_model"
``

Or start with [gunicorn](http://gunicorn.org/).

```
gunicorn --bind 0.0.0.0:8500 wsgi
```

Or run with `uwsgi`.

```
uwsgi --http 0.0.0.0:8500 -w wsgi

uwsgi --http 0.0.0.0:8501 -w wsgi --pyargv "--model_name hello"
```
