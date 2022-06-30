
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='PyTorchTools',
    version='0.1.1',
    description='Simple tools for pytorch',
    url='https://github.com/FlorianMuellerklein/PyTorchTrainer',
    author='Florian Muellerklein',
    author_email='f.muellerklein@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=requirements,
    zip_safe=False
)