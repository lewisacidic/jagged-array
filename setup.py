#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="jagged-array",
        version="0.0.1-dev.0",
        packages=find_packages(),
        author="Rich Lewis",
        author_email="opensource@richlew.is",
        description="Multidimensional [jr]agged array support for the PyData ecosystem",
        license="MIT",
        keywords="jagged ragged multidimensional array",
        url="https://github.com/lewisacidic/jagged",
        requirements=["numpy"],
        extras_require={
            "dev": [
                "pytest",
                "black",
                "flake8",
                "pytest-cov",
                "pytest-flake8",
                "pytest-cov"
            ]
        }
    )
