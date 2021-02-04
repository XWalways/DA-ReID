import torch
from opt import opt
from tqdm import tqdm
from model import ft_net
from scipy.spatial.distance import cdist
from torch.optim import lr_scheduler
from utils.metrics import mean_ap, cmc, re_ranking

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

model = ft_net(opt.num_cls, droprate=0.5, stride=1)
model.to(opt.device)
train_loader, train_loader_woEr, test_loader, query_loader, dataset = make_data_loader(opt)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=5e-4, amsgrad=True)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_scheduler, gamma=0.1)
cross_entropy_loss = torch.nn.CrossEntropyLoss()

for epoch in range(opt.start+1, opt.epoch+1):
    print('\nepoch', epoch)
    model.train()
    scheduler.step()
    for batch, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(opt.device)
        labels = labels.to(opt.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cross_entropy_loss(outputs[1], labels)
        loss.backward()
        optimizer.step()
        print('\rCE:%.2f' % (loss.data.cpu().numpy()), end=' ')

    if epoch % 2 == 0:
        model.eval()
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

        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
name = ''
if opt.data_path == 'Market-1501-v15.09.15':
    name += 'Market'
torch.save({'model': model.state_dict()}, 'model_{}.pth'.format(name))


