import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from contextlib import contextmanager
import pprint
import sys
from collections import Mapping, OrderedDict

from .query_expansion import aqe
from .rank import evaluate_rank



class ReidEvaluator(object):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(outputs.cpu())

    def evaluate(self):
        features = self.features
        pids = self.pids
        camids = self.camids

        features = torch.cat(features, dim=0)
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.aqe_test:
            qe_time = self.cfg.aqe_time
            qe_k = self.cfg.aqe_k
            alpha = self.cfg.aqe_alpha

            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        mAP = np.mean(all_AP)

        mINP = np.mean(all_INP)

        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100

        self._results['mAP'] = mAP * 100

        self._results['mINP'] = mINP * 100

        self._results['metric'] = (mAP + cmc[0]) / 2 * 100
        
        return copy.deepcopy(self._results)


def build_dist(feat_1, feat_2, metric = "euclidean", **kwargs):
    if metric == "euclidean":
        return compute_euclidean_distance(feat_1, feat_2)

    elif metric == "cosine":
        return compute_cosine_distance(feat_1, feat_2)

@torch.no_grad()
def compute_euclidean_distance(features, others):
    m, n = features.size(0), others.size(0)
    dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, features, others.t())

    return dist_m.cpu().numpy()


@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def inference_on_dataset(model, data_loader, evaluator, flip_test=False):
    total = len(data_loader)
    evaluator.reset()
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            outputs = model(inputs)
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
        evaluator.process(inputs, outputs)

    results = evaluator.evaluate()

    return results

