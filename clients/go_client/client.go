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
    dataByte := []byte(`{"data": {"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}}`)
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
