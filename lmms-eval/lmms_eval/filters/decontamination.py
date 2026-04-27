# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lmms_eval.api.filter import Filter


class DecontaminationFilter(Filter):
    """
    A filter which evaluates
    """

    name = "track_decontamination"

    def __init__(self, path) -> None:
        """

        TODO: make sure only ever run one time on the train set (should this be cached as a class var? keyed by value for "path").
        should further cache result on a given (task_name, doc_id)
        """
        self._decontam_results = None

    def apply(self, resps, docs) -> None:
        """
        Return {"no_contamination", "only_contamination"} keys for the 2 different subsets
        """
        pass
