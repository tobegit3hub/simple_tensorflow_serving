#!/usr/bin/env python

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import SparseVector

spark = SparkSession.builder.appName("libsvm_lr").getOrCreate()

# Load model
model_path = "./lr_model/"
lrModel = LogisticRegressionModel.load(model_path)
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# Construct data
#testset = spark.read.format("libsvm").load("./sample_libsvm_data.txt")
testset = spark.createDataFrame(
    [(1.0, SparseVector(692, [128, 129, 130], [51, 159, 20]))],
    ['label', 'features'])

# Make inference
result = lrModel.transform(testset)
result = result.first()
print("Prediction: {}, probability_of_0: {}, probability_of_1: {}".format(
    result.label, result.probability[0], result.probability[1]))
