import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np
#from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
from model import ft_net
from data import make_data_loader
from opt import opt

opt.data_path = '../DukeMTMC-reID'
train_loader, train_loader_woEr, test_loader, query_loader, dataset = make_data_loader(opt)
model = ft_net(class_num=702)
model.eval()
model.cuda()
model.load_state_dict(torch.load('./teacher_dukemtmc.pth'))


print('extract features, this may take a few minutes')
#qf = extract_feature(model, tqdm(query_loader)).numpy()
#gf = extract_feature(model, tqdm(test_loader)).numpy()

def extract_feature(model, loader):
    features = torch.FloatTensor()

    for (inputs, labels) in loader:
        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')
            outputs = model(input_img)
            f = outputs[0].data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features

qf = extract_feature(model, tqdm(query_loader)).numpy()
gf = extract_feature(model, tqdm(test_loader)).numpy()

def rank(dist):
    r = cmc(dist, dataset.query_pids, dataset.gallery_pids, dataset.query_cams, dataset.gallery_cams,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True)
    m_ap = mean_ap(
            dist, dataset.query_pids, dataset.gallery_pids, dataset.query_cams, dataset.gallery_cams)

    return r, m_ap

#########################   re rank##########################
q_g_dist = np.dot(qf, np.transpose(gf))
q_q_dist = np.dot(qf, np.transpose(qf))
g_g_dist = np.dot(gf, np.transpose(gf))
dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

r, m_ap = rank(dist)

print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        .format(m_ap, r[0], r[2], r[4], r[9]))

#########################no re rank##########################
dist = cdist(qf, gf)

r, m_ap = rank(dist)

print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
        .format(m_ap, r[0], r[2], r[4], r[9]))

