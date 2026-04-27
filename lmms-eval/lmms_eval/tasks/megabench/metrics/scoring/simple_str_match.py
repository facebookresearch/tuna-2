# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from metrics.scoring.exact_str_match import ExactStrMatch


class SimpleStrMatch:
    """Basic string matching, without spaces or hyphens."""

    @staticmethod
    def match(response, correct_answer: str) -> int:
        """Simple string match between response and correct_answer."""
        if not isinstance(response, str):
            response = str(response)  # If it is JSON-like
        response = response.replace(" ", "").replace("-", "").replace("\n", "").replace("\t", "").replace(".", "").lower()
        correct_answer = correct_answer.replace(" ", "").replace("-", "").replace("\n", "").replace("\t", "").replace(".", "").lower()

        return ExactStrMatch.match(response, correct_answer)
