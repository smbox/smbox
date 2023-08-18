from setuptools import setup, find_packages
import re

# Read the package version
with open("smbox/__init__.py", "r") as fh:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        fh.read(),
        re.MULTILINE
    ).group(1)

# Read the contents of the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='smbox',
    version=version,
    url='https://github.com/Tarek0/smbox',
    author='Tarek Salhi',
    author_email='tareksalhi0@gmail.com',
    license='MIT',
    description='A lightweight HPO package to efficiently optimize the hyperparameters of an ML algorithm.',
    packages=find_packages(),    
    install_requires=requirements,
)
