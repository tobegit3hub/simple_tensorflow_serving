import base64

from . import base64_util

test_data1 = {"inputs": {"images": {"b64": "YWJjZGVmZ2hpMTIz"}, "foo": "bar"}}
print(test_data1)
base64_util.replace_b64_in_dict(test_data1)
print(test_data1)

test_data1 = {
    "inputs": {
        "images": [{
            "b64": "YWJjZGVmZ2hpMTIz"
        }, "foo", "bar"]
    }
}
print(test_data1)
base64_util.replace_b64_in_dict(test_data1)
print(test_data1)

test_data1 = {
    "inputs": {
        "images": {
            "b64": "YWJjZGVmZ2hpMTIz"
        },
        "foo": [{
            "images": {
                "b64": "YWJjZGVmZ2hpMTIz"
            }
        }]
    }
}
print(test_data1)
base64_util.replace_b64_in_dict(test_data1)
print(test_data1)

test_data1 = {
    "inputs": {
        "images": [[{
            "b64": "YWJjZGVmZ2hpMTIz"
        }, "foo"], [{
            "b64": "YWJjZGVmZ2hpMTIz"
        }, "bar"]]
    }
}
print(test_data1)
base64_util.replace_b64_in_dict(test_data1)
print(test_data1)

input_data = {
    "model_name": "tensorflow_template_application_model",
    "model_version": 1,
    "signature_name": "serving_default",
    "data": {
        "keys": [[1.0], [2.0]],
        "b64_inputs": [[{
            "b64": "YWJjZGVmZ2hpMTIz"
        }], ["hello"]],
        "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]
    }
}
test_data1 = input_data
print(test_data1)
base64_util.replace_b64_in_dict(test_data1)
print(test_data1)

test_data1 = {
    'data': {
        'keys': [[1.0], [2.0]],
        'features': [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        'key': [[{
            'b64': u'YWJjZGVmZ2hpMTIz'
        }]]
    }
}
print(test_data1)
base64_util.replace_b64_in_dict(test_data1)
print(test_data1)


def generate_b64_string():
  test_string = "abcdefghi123"
  b64_string = base64.urlsafe_b64encode(test_string)
  print(b64_string)

  test_string = base64.urlsafe_b64decode(b64_string)
  import ipdb
  ipdb.set_trace()
  print(test_string)
