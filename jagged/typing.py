#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Rich Lewis <opensource@richlew.is>
# License: MIT

from typing import TextIO, BinaryIO, Union
from pathlib import Path

FileLike = Union[BinaryIO, TextIO, str, Path]
