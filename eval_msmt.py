import torch
import os
import numpy as np

from evaluation import evaluate_rank
from data import make_data_loader
from network import Model
from tqdm import tqdm
from opt import opt
def extract_feature(model, loader):
    features = torch.FloatTensor()

    for (inputs, labels) in loader:
        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')
            outputs = model.C(input_img)
            f = outputs[0].data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features

def build_dist(feat_1, feat_2):
    m, n = feat_1.size(0), feat_2.size(0)
    dist_m = (
            torch.pow(feat_1, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(feat_2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist_m.addmm_(1, -2, feat_1, feat_2.t())

    return dist_m.cpu().numpy()

model = Model()
train_loader, train_loader_woEr, test_loader, query_loader, dataset = make_data_loader(opt)
checkpoint = torch.load(os.path.join(opt.save_path, opt.weight))
model.C.load_state_dict(checkpoint['model_C'], strict=False)
model.cuda()
model.eval()
print('extract features, this may take a few minutes')
query_features = extract_feature(model, tqdm(query_loader))
gallery_features = extract_feature(model, tqdm(test_loader))

dist = build_dist(query_features, gallery_features)

cmc, all_AP, all_INP = evaluate_rank(dist, dataset.query_pids, dataset.gallery_pids, dataset.query_cams, dataset.gallery_cams)
mAP = np.mean(all_AP)
Rank1 = cmc[0]
print('MSMT17 mAP: {}, Rank-1: {}'.format(mAP, Rank1))
