# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from metrics.scoring.common.conversions import str_to_list
from metrics.scoring.common.metrics import longest_common_prefix


class LongestCommonListPrefixRatio:
    """Determines how much of the first part of the list
    was predicted correctly.
    """

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        responses = str_to_list(responses)
        targets = str_to_list(targets)
        return len(longest_common_prefix(responses, targets)) / len(targets)
