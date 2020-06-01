import os
from setuptools import setup, find_packages


setup(
    name='effective_transformer',
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    extras_require={
        'tensorflow': ['tensorflow==1.15.3'],
    },
)