# Custom Op

## Introduction

If you want to use the new [custom op](https://www.tensorflow.org/extend/adding_an_op), run the script to build the example.

```
./build_custom_op.sh
```

Now we can train and export the model with this script.

```
./main.py
```

# Run Server

If we run the serving service like this, we will get the error about missing custom op.

```
simple_tensorflow_serving --model_base_path="./model/"
```

Now we can load the custom op while starting the serving service with this command.

```
simple_tensorflow_serving --model_base_path="./model/" --custom_op_paths="./"
```
