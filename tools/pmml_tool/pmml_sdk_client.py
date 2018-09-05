#!/usr/bin/env python

from openscoring import Openscoring

os = Openscoring("http://localhost:8080/openscoring")

kwargs = {"auth": ("admin", "adminadmin")}

model_file_path = "../../models/pmml_iris/DecisionTreeIris.pmml"
model_name = "PmmlModel"
os.deployFile(model_name, model_file_path, **kwargs)

arguments = {
    "Sepal_Length": 5.1,
    "Sepal_Width": 3.5,
    "Petal_Length": 1.4,
    "Petal_Width": 0.2
}

result = os.evaluate(model_name, arguments)
print(result)
