from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "./README.md"), encoding="utf-8") as f:
    lines = f.readlines()
# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="mutex",
    packages=[
        package for package in find_packages() if (package.startswith("mutex"))
    ],
    install_requires=[],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="MUTEX: Learning Unified Policies from Multimodal Task Specifications",
    author="Rutav Shah, Roberto Martín-Martín, Yuke Zhu",
    author_email="rutavms@cs.utexas.edu",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
