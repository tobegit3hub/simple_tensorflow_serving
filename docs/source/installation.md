# Installation

## Pip

Install the server with [pip](https://pypi.python.org/pypi/simple-tensorflow-serving).

```bash
pip install simple_tensorflow_serving
```

## Source

Install from [source code](https://github.com/tobegit3hub/simple_tensorflow_serving).

```bash
git clone https://github.com/tobegit3hub/simple_tensorflow_serving

cd ./simple_tensorflow_serving/

python ./setup.py install
```

## Bazel

Install with [bazel](https://bazel.build/).

```bash
git clone https://github.com/tobegit3hub/simple_tensorflow_serving

cd ./simple_tensorflow_serving/

bazel build simple_tensorflow_serving:server
```

## Docker

Deploy with [docker image](https://hub.docker.com/r/tobegit3hub/simple_tensorflow_serving/).

```bash
docker run -d -p 8500:8500 tobegit3hub/simple_tensorflow_serving

docker run -d -p 8500:8500 tobegit3hub/simple_tensorflow_serving:latest-gpu

docker run -d -p 8500:8500 tobegit3hub/simple_tensorflow_serving:latest-hdfs

docker run -d -p 8500:8500 tobegit3hub/simple_tensorflow_serving:latest-py34
```

## Docker Compose

Deploy with [docker-compose](https://docs.docker.com/compose/).

```bash
wget https://raw.githubusercontent.com/tobegit3hub/simple_tensorflow_serving/master/docker-compose.yml

docker-compose up -d
```

## Kubernetes

Deploy in [Kubernetes](https://kubernetes.io/) cluster.

```bash
wget https://raw.githubusercontent.com/tobegit3hub/simple_tensorflow_serving/master/simple_tensorflow_serving.yaml

kubectl create -f ./simple_tensorflow_serving.yaml
```