#!/usr/bin/env swift


import Foundation


var str = "Hello, playground"


print("Run swift code")





var url2: NSURL = NSURL(string: "http://url.com/path/to/file.php")!
var request:NSMutableURLRequest = MutableURLRequest(URL:url2)


var bodyData = "data=something"
request.HTTPMethod = "POST"



request.HTTPBody = bodyData.dataUsingEncoding(NSUTF8StringEncoding);
NSURLConnection.sendAsynchronousRequest(request, queue: NSOperationQueue.mainQueue())
{
    (response, data, error) in
    println(response)
    
}



// prepare json data
let json: [String: Any] = ["title": "ABC", "dict": ["1":"First", "2":"Second"]]

//let json: [String: Any] = ["keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]]

let jsonData = try? JSONSerialization.data(withJSONObject: json)
//print(jsonData)


let url = URL(string: "http://127.0.0.1:8500")!
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.httpBody = jsonData

let task = URLSession.shared.dataTask(with: request) { data, response, error in
    guard let data = data, error == nil else {
        print(error?.localizedDescription ?? "No data")
        return
    }
    
    
    let responseJSON = try? JSONSerialization.jsonObject(with: data, options: [])
    
    
    if let responseJSON = responseJSON as? [String: Any] {
        print(responseJSON)
    }
}

task.resume()



print("End")

