#!/usr/bin/env node

var request = require('request');

var options = {
    uri: "http://127.0.0.1:8500",
    method: "POST",
    json: {"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}
};

request(options, function (error, response, body) {
    if (!error && response.statusCode == 200) {
        console.log(body)
    } else {
        console.log(error)
    }
});