import logging
import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from contextlib import contextmanager

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

        if cfg.rerank:
            k1 = cfg.rerank_k1
            k2 = cfg.rerank_k2
            lamda_value = self.cfg.rerank_lamda

            if self.cfg.test_metric == 'cosine':
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

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

def aqe(query_feat, gallery_feat):
    num_query = query_feat.shape[0]
    all_feat = torch.cat((query_feat, gallery_feat), dim=0)
    norm_feat = F.normalize(all_feat, p=2, dim=1)

    all_feat = all_feat.numpy()
    for i in range(qe_times):
        all_feat_list = []
        sims = torch.mm(norm_feat, norm_feat.t())
        sims = sims.data.cpu().numpy()
        for sim in sims:
            init_rank = np.argpartition(-sim, range(1, qe_k + 1))
            weights = sim[init_rank[:qe_k]].reshape((-1, 1))
            weights = np.power(weights, alpha)
            all_feat_list.append(np.mean(all_feat[init_rank[:qe_k], :] * weights, axis=0))
        all_feat = np.stack(all_feat_list, axis=0)
        norm_feat = F.normalize(torch.from_numpy(all_feat), p=2, dim=1)

    query_feat = torch.from_numpy(all_feat[:num_query])
    gallery_feat = torch.from_numpy(all_feat[num_query:])
    return query_feat, gallery_feat

def evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, g_pids, q_camids, g_camids, max_rank)
    else:
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)

def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    num_repeats = 10

    num_q, num_g = distmat.shape

    indices = np.argsort(distmat, axis=1)

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
                format(num_g)
        )

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP

def inference_on_dataset(model, data_loader, evaluator, flip_test=False):
    total = len(data_loader)
    evaluator.reset()

    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            outputs = model(inputs)
            # Flip test
            if flip_test:
                inputs["images"] = inputs["images"].flip(dims=[3])
                flip_outputs = model(inputs)
                outputs = (outputs + flip_outputs) / 2
            evaluator.process(inputs, outputs)
    results = evaluator.evaluate()
    if results is None:
        results = {}
    return results

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
