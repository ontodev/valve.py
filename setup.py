from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(here + "/README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["lark-parser"]

test_requires = []

setup(
    name="ontodev-valve",
    description="VALVE Spreadsheet Validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    author="Rebecca Jackson",
    author_email="rbca.jackson@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=install_requires,
    setup_requires=["pytest-runner"],
    packages=find_packages(exclude="tests"),
    entry_points={"console_scripts": ["valve=valve.valve:main"]},
)
