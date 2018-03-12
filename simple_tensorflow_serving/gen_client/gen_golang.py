import json
import logging

from jinja2 import Template


def gen_tensorflow_client(generated_tensor_data):
  """
  Generate TensorFlow SDK in Golang.

  Args:
    generated_tensor_data: Example is {"keys": [[1.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
  """

  code_template = """
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "bytes"
    "net/http"
    "io/ioutil"
)

func main() {
    endpoint := "http://127.0.0.1:8500"
    log.Print("Request tensorflow serving in " + endpoint)

    // Construct request data
    dataByte := []byte(`{"data": {{ tensor_data }} }`)
    var dataInterface map[string]interface{}
    json.Unmarshal(dataByte, &dataInterface)
    dataJson, _ := json.Marshal(dataInterface)

    // Send POST request
    resp, err := http.Post(endpoint, "application/json", bytes.NewBuffer(dataJson))

    // Print the response
    if err != nil {
        log.Print("Error to request server")
        return

    }
    defer resp.Body.Close()
    body, err := ioutil.ReadAll(resp.Body)
    fmt.Println(string(body))
}
  """

  generated_tensor_data_string = json.dumps(generated_tensor_data)
  template = Template(code_template)
  generate_code = template.render(tensor_data=generated_tensor_data_string)
  logging.debug("Generate the code in Golang:\n{}".format(generate_code))

  generated_code_filename = "client.go"
  with open(generated_code_filename, "w") as f:
    f.write(generate_code)

  logging.info('Save the generated code in {}, try "go run {}"'.format(
      generated_code_filename, generated_code_filename))
