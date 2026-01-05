# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

from collections import OrderedDict

import numpy as np
import torch

from recbole.evaluator.collector import DataStruct
from recbole.evaluator.register import metrics_dict


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
        return result_dict


class FA_IREvaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics."""

    def __init__(self, config):
        print("Using FA_IR evaluator")
        self.config = config
        self.metrics = [metric.lower() for metric in self.config["metrics"]]
        self.metric_class = {}
        self._protected_map = None
        self.tail_ratio = config["tail_ratio"]

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

    def _build_protected_map(self, dataobject):
        # TODO Move to trainer
        # DITTO the other fa_ir code.
        from collections import defaultdict

        count_items = dataobject.get("data.count_items").most_common()

        total = 0
        for item, count in count_items:
            total += count

        protected_map = defaultdict(lambda: False)
        covered_prob = 0.0  # swap with a different condition
        for idx in range(len(count_items) - 1, -1, -1):
            item, count = count_items[idx]
            prob = count / total
            covered_prob += prob

            if covered_prob >= self.tail_ratio:
                break

            protected_map[item] = True

        return protected_map

    def protected_map(self, dataobject):
        if not self._protected_map:
            self._protected_map = self._build_protected_map(dataobject)

        return self._protected_map

    def evaluate(self, dataobject: DataStruct):
        """calculate all the metrics. It is called at the end of each epoch

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: such as ``{'hit@20': 0.3824, 'recall@20': 0.0527, 'hit@10': 0.3153, 'recall@10': 0.0329, 'gauc': 0.9236}``

        """
        self.apply_fa_ir(dataobject)

        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
        return result_dict
