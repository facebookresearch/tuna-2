# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import rapidfuzz


class NormalizedSimilarityDamerauLevenshtein:
    """Normalized Damerau-Levenshtein Similarity."""

    @staticmethod
    def match(response, correct_answer) -> int:
        """Normalized indel similarityuiio do between targets and responses."""
        if not isinstance(response, str) and isinstance(correct_answer, str):
            return 0
        return rapidfuzz.distance.DamerauLevenshtein.normalized_similarity(response, correct_answer)
