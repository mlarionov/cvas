from setuptools import setup, find_packages
from os import path

__version__ = '0.0.2'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name = "cvas",
    version = __version__,
    author = "Michael Larionov",
    description ="Cross-validation framework",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license = "MIT",
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'optuna',
        'scikit-learn',
        'pandas'
    ]
)