# Update the code and upload the package to pypi
# 1. python ./setup.py bdist_wheel --universal
# 2. twine upload dist/simple_tensorflow_serving-x.x.x-py2.py3-none-any.whl

from setuptools import setup, find_packages

setup(
    name="simple_tensorflow_serving",
    version="0.6.6",
    author="tobe",
    author_email="tobeg3oogle@gmail.com",
    url="https://github.com/tobegit3hub/simple_tensorflow_serving",
    description=
    "The simpler and easy-to-use serving service for TensorFlow models",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'protobuf',
        'onnx',
        'flask',
        'jinja2',
        'flask-cors',
        'requests',
        'pillow',
        'tensorflow'
    ],
    #package_data={
    #    "simple_tensorflow_serving/static": ['simple_tensorflow_serving/templates/*.html',
    #                                         'simple_tensorflow_serving/static/*']
    #},
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "simple_tensorflow_serving=simple_tensorflow_serving.server:main",
        ],
    })
