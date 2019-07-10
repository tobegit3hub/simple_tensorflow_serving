#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import configparser
import subprocess


def print_usage():
  """
  Print the usage of this class.
  """

  print('Usage: simple_tensorflow_serving --model_base_path="./model"')


def update_uwsgi_conf(args, uwsgi_conf):
  host = os.environ.get("STFS_HOST", "0.0.0.0")
  port = int(os.environ.get("STFS_PORT", "8500"))

  for arg in args:
    if arg.startswith("-h") or arg.startswith("--help"):
      print_usage()
      return
    if arg.startswith("--host"):
      host = arg[7:]
      print("Use the host: {}".format(host))
    if arg.startswith("--port"):
      port = int(arg[7:])
      print("Use the port: {}".format(port))

    worker_number = int(os.environ.get("STFS_WORKERS", "1"))
    if arg.startswith("--workers"):
      worker_number = int(arg[10:])

    thread_number = int(os.environ.get("STFS_THREADS", "1"))
    if arg.startswith("--threads"):
        thread_number = int(arg[10:])

  uwsgi_conf["uwsgi"]["http"] = "{}:{}".format(host, port)
  uwsgi_conf["uwsgi"]["workers"] = worker_number
  uwsgi_conf["uwsgi"]["threads"] = thread_number


def main():
  """
  Start new uwsgi progress for simple tensorflow serving.
  """

  # 1. Parse command-line parameter to generate uwsgi conf
  args = sys.argv[1:]
  uwsgi_conf = {
      "uwsgi": {
          "module": "simple_tensorflow_serving.server:app",
          "pyargv": " ".join(args),
          "http": "0.0.0.0:8500",
          "socket": "/tmp/uwsgi.sock",
          "pidfile": "/tmp/uwsgi.pid",
          "master": True,
          "close-on-exec": True,
          "enable-threads": True,
          "http-keepalive": 1,
          "http-auto-chunked": 1,
          "workers": 1,
          "threads": 1,
          # TODO: Log format refers to https://uwsgi-docs.readthedocs.io/en/latest/LogFormat.html
          #"log-format": '%(ltime) "%(method) %(uri) %(proto)" %(status) %(size) "%(referer)" "%(uagent)"'
      }
  }
  update_uwsgi_conf(args, uwsgi_conf)
  print("Uwsgi config: {}".format(uwsgi_conf))

  # 2. Save config file of uwsgi.ini
  uwsgi_ini_file = "/tmp/uwsgi.ini"
  with open(uwsgi_ini_file, "w") as f:
    uwsgi_confparser = configparser.ConfigParser()
    uwsgi_confparser.read_dict(uwsgi_conf)
    uwsgi_confparser.write(f)
  print("Save uwsgi config in: {}".format(uwsgi_ini_file))

  # 3. Start uwsgi command
  uwsgi_command = "uwsgi --ini {}".format(uwsgi_ini_file)
  print("Try to run command: {}".format(uwsgi_command))
  subprocess.call(uwsgi_command, shell=True)


if __name__ == "__main__":
  main()
