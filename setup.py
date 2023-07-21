# Allows for pip install -e .

from setuptools import setup, find_packages

setup(
    name='lippy',
    version='0.1',
    packages=find_packages(),
)