FROM python:2.7

ADD ./requirements.txt /

RUN pip install -r /requirements.txt

RUN pip install simple-tensorflow-serving

ADD . /
WORKDIR /

EXPOSE 8500

CMD ['simple_tensorflow_serving', '--port=8500', '--model_base_path="./saved_model/1"']