# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import yaml


def load_default_template_yaml(task_file):
    with open(Path(task_file).parent / "_default_template_yaml", "r") as f:
        safe_data = [line for line in f if "!function" not in line]
    return yaml.safe_load("".join(safe_data))
