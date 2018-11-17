#!/usr/bin/env Rscript

library(httr)

main <- function(){
  endpoint <- "http://127.0.0.1:8500"
  body <- list(data = list(a = 1), keys = 1)

  json_data <- list(
    data = list(
      keys = list(list(1.0), list(2.0)), features = list(list(1, 1, 1, 1, 1, 1, 1, 1, 1), list(1, 1, 1, 1, 1, 1, 1, 1, 1))
    )
  )

  r <- POST(endpoint, body = json_data, encode = "json")

  stop_for_status(r)

  # print(r)
  content(r, "parsed", "text/html")
}

main()
