# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class DummyParse:
    @staticmethod
    def parse(response: str, *args, **kwargs) -> dict:
        """return the raw string without doing anything"""
        return response.strip()
