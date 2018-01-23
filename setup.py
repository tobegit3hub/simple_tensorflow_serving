try:
  from setuptools import setup
  setup()
except ImportError:
  from distutils.core import setup

setup(
    name="simple_tensorflow_serving",
    version="0.1.1",
    author="tobe",
    author_email="tobeg3oogle@gmail.com",
    url="https://github.com/tobegit3hub/simple_tensorflow_serving",
    install_requires=["tensorflow>=1.0.0"],
    description=
    "The simpler and easy-to-use serving service for TensorFlow models",
    packages=["simple_tensorflow_serving"],
    entry_points={
        "console_scripts": [
            "simple_tensorflow_serving=simple_tensorflow_serving.server:main",
        ],
    })
