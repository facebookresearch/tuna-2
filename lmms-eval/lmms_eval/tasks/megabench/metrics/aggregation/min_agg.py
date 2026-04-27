# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from numbers import Number
from typing import Dict


class MinAggregation:
    """Take the minimum of all valid scores."""

    @staticmethod
    def aggregate(scores: Dict[str, Number], weights: Dict[str, Number]) -> Number:
        """Exact match between targets and responses."""
        filtered_scores = [s for s in scores.values() if s >= 0]
        if not filtered_scores:
            return -1
        return min(filtered_scores)
