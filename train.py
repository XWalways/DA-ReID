import torch
from opt import opt
from tqdm import tqdm
from model import ft_net
from scipy.spatial.distance import cdist
from torch.optim import lr_scheduler
from utils.metrics import mean_ap, cmc, re_ranking

opt.epoch = 60
model = ft_net(opt.num_cls, droprate=0.5, stride=1)
model.cuda()
train_loader, train_loader_woEr, test_loader, query_loader, dataset = make_data_loader(opt)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=5e-4, amsgrad=True)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[opt.epoch], gamma=0.1)
cross_entropy_loss = torch.nn.CrossEntropyLoss()

for epoch in range(opt.start+1, opt.epoch+1):
    print('\nepoch', epoch)
    model.train()
    scheduler.step()
    for batch, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cross_entropy_loss(outputs[1], labels)
        loss.backward()
        optimizer.step()
        print('\rCE:%.2f' % (loss.data.cpu().numpy()), end=' ')


name = ''
if opt.data_path == 'Market-1501-v15.09.15':
    name += 'market'
if opt.data_path == 'DukeMTMC-reID':
    name += 'dukemtmc'
if opt.data_path == 'cuhk03':
    name += 'cuhk03'
torch.save(model.cpu().state_dict(), 'teacher_{}.pth'.format(name))


