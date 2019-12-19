import os

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [l.strip() for l in f]

tests_require = ["pytest", "pytest-cov"]


setup(
    name="wildml",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
)
