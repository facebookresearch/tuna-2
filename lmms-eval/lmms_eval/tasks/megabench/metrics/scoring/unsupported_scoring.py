# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

class UnsupportedScoring:
    """Unsupported scoring."""

    @staticmethod
    def match(response: str, correct_answer: str) -> int:
        """Default response for unimplemented metrics."""
        return -1
