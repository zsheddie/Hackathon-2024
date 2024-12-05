# setup.py
import os
import setuptools

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup_py_dir = os.path.dirname(os.path.realpath(__file__))

setuptools.setup(
    name='ProKI Hackathon 2024',
    version='1.0',
    author='Max Goebels',
    description='ProKI Hackathon 2024 Example Project',
    install_requires=[req for req in requirements if req[:2] != "# "],
    packages=setuptools.find_packages()
)