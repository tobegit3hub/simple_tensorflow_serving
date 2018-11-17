FROM python:2.7

RUN apt-get update -y
RUN apt-get install -y unzip wget
RUN apt-get install -y default-jdk

RUN wget http://h2o-release.s3.amazonaws.com/h2o/rel-wolpert/8/h2o-3.18.0.8.zip
RUN unzip ./h2o-3.18.0.8.zip
RUN mv h2o-3.18.0.8/h2o.jar /tmp/

ADD ./requirements.txt /
RUN pip install -r /requirements.txt

ADD . /simple_tensorflow_serving/
WORKDIR /simple_tensorflow_serving/
RUN cp ./third_party/openscoring/openscoring-server-executable-1.4-SNAPSHOT.jar /tmp/

# RUN pip install simple-tensorflow-serving
RUN python ./setup.py install

EXPOSE 8500

# CMD ["simple_tensorflow_serving", "--port=8500", "--model_base_path=./models/tensorflow_template_application_model"]
CMD ["simple_tensorflow_serving", "--port=8500", "--model_config_file=./examples/model_config_file.json"]
