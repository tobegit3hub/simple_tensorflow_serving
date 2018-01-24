#!/usr/bin/env escript

main(_) ->
  ssl:start(),
  application:start(inets),
  httpc:request(post,
    {"http://127.0.0.1:8500", [],
    "application/json",
    "{\"keys\": [[11.0], [2.0]], \"features\": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}"
    }, [], []).
