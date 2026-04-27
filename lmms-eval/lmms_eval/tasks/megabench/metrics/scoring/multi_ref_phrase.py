# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from numbers import Number

from metrics.scoring.common.conversions import str_to_iterable
from metrics.scoring.simple_str_match import SimpleStrMatch


def replace_potential_chinese_comma(input_string):
    return input_string.replace("，", ",")


class MultipleReferencePhraseEval:
    """
    Check the response with multiple correct references
    As long as one is matched, the score is 1, otherwise the score is 0
    """

    @staticmethod
    def match(response, targets) -> Number:
        targets = replace_potential_chinese_comma(targets)
        refs = str_to_iterable(list, targets)
        matched = False
        for ref in refs:
            str_ref = ref if isinstance(ref, str) else str(ref)
            if SimpleStrMatch.match(response, str_ref):
                matched = True
                break
        return 1 if matched else 0
