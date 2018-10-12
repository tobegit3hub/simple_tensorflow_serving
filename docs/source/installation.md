## Installation

Install the server with [pip](https://pypi.python.org/pypi/simple-tensorflow-serving).

```
pip install simple_tensorflow_serving
```

Or install from [source code](https://github.com/tobegit3hub/simple_tensorflow_serving).

```
python ./setup.py install

python ./setup.py develop

bazel build simple_tensorflow_serving:server
```

Or use the [docker image](https://hub.docker.com/r/tobegit3hub/simple_tensorflow_serving/).

```
docker run -d -p 8500:8500 tobegit3hub/simple_tensorflow_serving

docker run -d -p 8500:8500 tobegit3hub/simple_tensorflow_serving:latest-gpu

docker run -d -p 8500:8500 tobegit3hub/simple_tensorflow_serving:latest-hdfs

docker run -d -p 8500:8500 tobegit3hub/simple_tensorflow_serving:latest-py34
```

````
docker-compose up -d
````

Or deploy in [Kubernetes](https://kubernetes.io/).

```
kubectl create -f ./simple_tensorflow_serving.yaml
```