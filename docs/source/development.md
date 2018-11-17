# Development

## Principle

1. `simple_tensorflow_serving` starts the HTTP server with `flask` application.
2. Load the TensorFlow models with `tf.saved_model.loader` Python API.
3. Construct the feed_dict data from the JSON body of the request.
   ```
   // Method: POST, Content-Type: application/json
   {
     "model_version": 1, // Optional
     "data": {
       "keys": [[1], [2]],
       "features": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]
     }
   }
   ```
4. Use the TensorFlow Python API to `sess.run()` with feed_dict data.
5. For multiple versions supported, it starts independent thread to load models.
6. For generated clients, it reads user's model and render code with [Jinja](http://jinja.pocoo.org/) templates. 

## Debug

You can install the server with `develop` and test when code changes.

```bash
git clone https://github.com/tobegit3hub/simple_tensorflow_serving

cd ./simple_tensorflow_serving/

python ./setup.py develop
```