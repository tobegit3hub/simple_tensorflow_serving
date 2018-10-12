#!/usr/bin/env ruby
# -*- coding: utf-8 -*-

require 'net/http'
require 'json'

def main()

  # Create post request
  endpoint = "http://127.0.0.1:8500"
  uri = URI.parse(endpoint)
  header = {"Content-Type" => "application/json"}
  input_data = {"data" => {"keys"=> [[11.0], [2.0]], "features"=> [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}}
  http = Net::HTTP.new(uri.host, uri.port)
  request = Net::HTTP::Post.new(uri.request_uri, header)
  request.body = input_data.to_json

  # Send post request
  response = http.request(request)
  puts response.body

end


if __FILE__ == $0
  main
end