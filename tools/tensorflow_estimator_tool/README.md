
# TensorFlow Estimator Tool

## Usage

Define the raw data in plain file. The first line should be keys and second line should be types. The following lines could be the data which is sperated with specified symbol such as spaces or comma.

```
a b
string float
你好 -1
hello 0.5
```

The run the script to generate the string data.

```
./generate_estimator_string.py
```
