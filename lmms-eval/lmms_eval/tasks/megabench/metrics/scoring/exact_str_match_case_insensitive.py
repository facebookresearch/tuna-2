# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from metrics.scoring.exact_str_match import ExactStrMatch


class ExactStrMatchCaseInsensitive:
    """Case-insensitive exact string matching."""

    @staticmethod
    def match(response, correct_answer) -> int:
        """Case-insensitive exact match between targets and responses."""
        if not isinstance(response, str) and isinstance(correct_answer, str):
            return 0
        return ExactStrMatch.match(response.lower(), correct_answer.lower())
