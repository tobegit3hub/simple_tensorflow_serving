#!/usr/bin/env python

import requests


def main():
  model_name = "PmmlModel"
  endpoint = "http://localhost:8080/openscoring/model/{}".format(model_name)
  model_file_path = "../../models/pmml_iris/DecisionTreeIris.pmml"

  with open(model_file_path, "rb") as f:
    kwargs = {
        'headers': {
            'content-type': 'application/xml'
        },
        'json': None,
        'data': f,
        'auth': ('admin', 'adminadmin')
    }
    result = requests.put(endpoint, **kwargs)
    print("Deploy the model to Openscoring: {}".format(result))
    print("Deploy the model to Openscoring: {}".format(result.text))


if __name__ == "__main__":
  main()
