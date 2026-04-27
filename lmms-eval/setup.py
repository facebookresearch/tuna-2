# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools
from setuptools import setup

# This is to make sure that the package supports editable installs
if __name__ == "__main__":
    setuptools.setup(
        license_files=["LICENSE"],
    )
