#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from server import application

def example(env, start_response):
    start_response('200 OK', [('Content-Type','text/html')])
    return [b"Hello World"]

def main():
  application.run()

if __name__ == "__main__":
  main()
