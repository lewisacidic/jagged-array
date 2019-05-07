#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT
import shlex
import sys

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.test import test as TestCommand


class Flake8Command(Command):
    """ Command to lint code with Flake8 """

    user_options = [("flake8-args-", "a", "Arguments to pass to flake8")]

    def initialize_options(self):
        self.flake_args = ""

    def finalize_options(self):
        pass

    def run(self):
        from flake8.main.cli import main

        main(shlex.split(self.flake_args))


class BlackCommand(Command):
    """ Command to format code with Black """

    user_options = [("black-args-", "a", "Arguments to pass to black")]

    def initialize_options(self):
        self.black_args = "."

    def finalize_options(self):
        pass

    def run(self):
        import black

        black.main(shlex.split(self.black_args))


class PyTestCommand(TestCommand):
    """ command to run tests using pytest """

    user_options = [("pytest-args-", "a", "arguments to pass to pytest")]

    def initialize_options(self):
        super().initialize_options()
        self.pytest_args = ""

    def run_tests(self):
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


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
        install_requires=["numpy"],
        tests_require=["pytest", "pytest-cov", "pytest-flake8"],
        extras_require={
            "dev": [
                "pytest",
                "black",
                "flake8",
                "pytest-cov",
                "pytest-flake8",
                "pytest-cov",
            ]
        },
        cmdclass={"test": PyTestCommand, "format": BlackCommand, "lint": Flake8Command},
    )
