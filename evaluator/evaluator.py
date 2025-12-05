# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

import math
import torch
from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict
import numpy as np
from scipy.stats import binom


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
        from collections import defaultdict
        count_items = dataobject.get("data.count_items").most_common()
        
        total = 0
        for item, count in count_items:
            total += count
            
        protected_map = defaultdict(lambda: False)
        covered_prob = 0.0 # swap with a different condition
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
        
    def fa_ir(self, k: int, q: list[int], g: dict[int, bool], p: float, a: float) -> list[any]:
        """ TODO update desc
        Run FA*IR on a list of existing recommendations. Proudly lifted from: https://arxiv.org/pdf/1706.06368 .

        Args:
            k (int): Amount of items to select for recommendations.
            q_g (list[tuple[float, int, int]]): Relevance and protected status for every element.
            p (float): Minimum proportion of protected items
            a (float): Significance for each fair representation set

        Returns:
            list[str]: A list of recommendations.
        """
        res = [None for _ in range(k)]
        p_0, p_1 = [], []

        for entry in q:
            if g[entry]:
                p_1.append(entry)
            else: 
                p_0.append(entry)

        m = []
        for i in range(1, k + 1):
            min_prot = math.ceil(binom.ppf(a, i, p))
            m.append(min_prot)

        t_p, t_n = 0, 0
        while t_p + t_n < k:
            if t_p < m[t_p + t_n]:
                t_p += 1
                res[t_p + t_n - 1] = p_1.pop(0)
            else:
                p_1_el = p_1[0]
                p_0_el = p_0[0]

                if p_1_el >= p_0_el:
                    t_p += 1
                    res[t_p + t_n - 1] = p_1.pop(0)
                else:
                    t_n += 1
                    res[t_p + t_n - 1] = p_0.pop(0)

        return res

    def apply_fa_ir(self, dataobject: DataStruct):
        q_g = self.protected_map(dataobject)
        rec_items = dataobject.get("rec.items")
        rec_items = rec_items.numpy()
        n_users, K = rec_items.shape
        
        reranked_items = np.zeros((n_users, 10))
        for u in range(n_users):
            orig_rank = rec_items[u].tolist()
            new_rank = self.fa_ir(10, orig_rank, q_g, 0.3, 0.1) # insert fa_ir here
            reranked_items[u] = np.array(new_rank)

        reranked_items = torch.Tensor(reranked_items)
        dataobject.set("rec.items", reranked_items.clone())
        # dataobject.set("rec.topk", reranked_items.clone()) # 
        
        return dataobject

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
