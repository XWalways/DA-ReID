from torch.nn import CrossEntropyLoss, BCELoss, L1Loss, Tanh
from torch.nn.modules import loss
from utils.get_optimizer import get_optimizer
import torch
import torch.nn as nn
from torch.distributions import normal
import numpy as np
import copy
from model import ft_net
from opt import opt

batch_size = opt.batchid * opt.batchimage
num_gran = 8

class Loss(loss._Loss):
    def __init__(self, model):
        super(Loss, self).__init__()
        
        self.tanh = Tanh()
        self.l1_loss = L1Loss()
        self.bce_loss = BCELoss()
        self.cross_entropy_loss = CrossEntropyLoss()
        
        self.center_loss = CenterLoss(num_classes=opt.num_cls)
        self.teacher = ft_net(opt.num_cls, droprate=0.5, stride=1)
        self.teacher.cuda()
        self.teacher.load_state_dict(torch.load('teacher.pth'))
        self.model = model
        
        self.optimizer, self.optimizer_D = get_optimizer(model)
        
    def get_positive_pairs(self):
        idx=[]
        for i in range(batch_size):
            r = i
            while r == i:
                r = int(torch.randint(
                        low=opt.batchid*(i//opt.batchid), high=opt.batchid*(i//opt.batchid+1),
                        size=(1,)).item())
            idx.append(r)
        return idx
    
    def region_wise_shuffle(self, id, ps_idx):
        sep_id = id.clone()
        idx = torch.tensor([0]*(num_gran))
        while (torch.sum(idx)==0) and (torch.sum(idx)==num_gran):
            idx = torch.randint(high=2, size=(num_gran,))
        
        for i in range(num_gran):
            if idx[i]:
                sep_id[:, opt.feat_id*i:opt.feat_id*(i+1)] = id[ps_idx][:, opt.feat_id*i:opt.feat_id*(i+1)]
        return sep_id
    
    def get_noise(self):
        return torch.randn(batch_size, opt.feat_niz, device=opt.device)
    
    def make_onehot(self, label):
        onehot_vec = torch.zeros(batch_size, opt.num_cls)
        for i in range(label.size()[0]):
            onehot_vec[i, label[i]] = 1
        return onehot_vec
    
    def set_parameter(self, m, train=True):
        if train:
            for param in m.parameters():
                param.requires_grad = True
            m.apply(self.set_bn_to_train)
        else:
            for param in m.parameters():
                param.requires_grad = False
            m.apply(self.set_bn_to_eval)
    
    def set_bn_to_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.eval()
        
    def set_bn_to_train(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            m.train()
    
    def set_model(self):
        self.model.C.zero_grad()
        self.model.G.zero_grad()
        self.model.D.zero_grad()
        
        if opt.stage == 1:
            self.set_parameter(self.model.C, train=True)
            nid_dict1 = self.model.C.get_modules(self.model.C.nid_dict1())
            nid_dict2 = self.model.C.get_modules(self.model.C.nid_dict2())
            for i in range(np.shape(nid_dict1)[0]):
                self.set_parameter(nid_dict1[i], train=False)
            for i in range(np.shape(nid_dict2)[0]):
                self.set_parameter(nid_dict2[i], train=False)
            self.set_parameter(self.model.G, train=False)
            self.set_parameter(self.model.D, train=False)
            
        elif opt.stage == 2:
            self.set_parameter(self.model.C, train=False)
            nid_dict1 = self.model.C.get_modules(self.model.C.nid_dict1())
            nid_dict2 = self.model.C.get_modules(self.model.C.nid_dict2())
            for i in range(np.shape(nid_dict1)[0]):
                self.set_parameter(nid_dict1[i], train=True)
            for i in range(np.shape(nid_dict2)[0]):
                self.set_parameter(nid_dict2[i], train=True)
            self.set_parameter(self.model.G, train=True)
            self.set_parameter(self.model.D, train=True)
    
    def id_related_loss_1(self, labels, outputs, outputs_teacher, kl_type='avgloss'):
        #in stage 1, use KL, kl_type is avgloss or avgvec
        #in stage 1, the better result is: labelsmooth, Triplet + Center + CE, Triplet + CE, Triplet + labelsmooth
        if opt.labelsmooth:
            CrossEntropy_Loss = [CrossEntropyLabelSmooth(num_classes=opt.num_cls)(output, labels) for output in outputs[1:1+num_gran]]
        else:
            CrossEntropy_Loss = [self.cross_entropy_loss(output, labels) for output in outputs[1:1+num_gran]]
        Loss =  sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)
        if kl_type == 'avgloss':
            kl_loss = nn.KLDivLoss(reduction='sum')
            KD_Loss = [kl_loss(output, outputs_teacher[1])/output.size(0) for output in outputs[1:1+num_gran]]
            Loss += 0.0001*sum(KD_Loss) / len(KD_Loss)
        if kl_type == 'avgvec':
            mean_output = sum(outputs[1:1+num_gran])/num_gran
            KD_Loss = kl_loss(mean_output, outputs_teacher[1])/mean_output.size(0)
            Loss += 0.0001*KD_Loss
        if opt.triplet:
            Triplet_Loss = TripletLoss()(outputs[0], labels)[0]
            Loss += Triplet_Loss
        if opt.center:
            Center_Loss = self.center_loss(outputs[0], labels)
            Loss += 0.001*Center_Loss

        return Loss

    def id_related_loss_3(self, labels, outputs):
        #in stage 3 do not use KL,or other tricks
        CrossEntropy_Loss = [self.cross_entropy_loss(output, labels) for output in outputs[1:1+num_gran]]
        return sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

    def KL_loss(self, outputs):
        list_mu = outputs[-3]
        list_lv = outputs[-2]
        loss_KL = 0.
        for i in range(np.size(list_mu)):
            loss_KL += torch.sum(0.5 * (list_mu[i]**2 + torch.exp(list_lv[i]) - list_lv[i] - 1))
        return loss_KL/np.size(list_mu)
    
    def GAN_loss(self, inputs, outputs, labels):
        id = outputs[0]
        nid = outputs[-1]
        one_hot_labels = self.make_onehot(labels).to(opt.device)
        
        # Auto Encoder
        auto_G_in = torch.cat((id, nid, self.get_noise()), dim=1)
        auto_G_out = self.model.G.forward(auto_G_in, one_hot_labels)
        
        # Positive Shuffle
        ps_idx = self.get_positive_pairs()
        ps_G_in = torch.cat((id[ps_idx], nid, self.get_noise()), dim=1)
        ps_G_out = self.model.G.forward(ps_G_in, one_hot_labels)
        
        # Separate Positive Shuffle
        sep_id = self.region_wise_shuffle(id, ps_idx)
        sep_G_in = torch.cat((sep_id, nid, self.get_noise()), dim=1)
        sep_G_out = self.model.G.forward(sep_G_in, one_hot_labels)
        
        ############################################## D_loss ############################################
        D_real, C_real = self.model.D(inputs)
        REAL_LABEL = torch.FloatTensor(D_real.size()).uniform_(0.7, 1.0).to(opt.device)
        D_real_loss = self.bce_loss(D_real, REAL_LABEL)
        C_real_loss = self.cross_entropy_loss(C_real, labels)
                
        auto_D_fake, auto_C_fake = self.model.D(auto_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(auto_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        auto_D_fake_loss = self.bce_loss(auto_D_fake, FAKE_LABEL)
        auto_C_fake_loss = self.cross_entropy_loss(auto_C_fake, labels)
        
        ps_D_fake, ps_C_fake = self.model.D(ps_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(ps_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        ps_D_fake_loss = self.bce_loss(ps_D_fake, FAKE_LABEL)
        ps_C_fake_loss = self.cross_entropy_loss(ps_C_fake, labels)
        
        sep_D_fake, sep_C_fake = self.model.D(sep_G_out.detach())
        FAKE_LABEL = torch.FloatTensor(sep_D_fake.size()).uniform_(0.0, 0.3).to(opt.device)
        sep_D_fake_loss = self.bce_loss(sep_D_fake, FAKE_LABEL)
        sep_C_fake_loss = self.cross_entropy_loss(sep_C_fake, labels)
                
        D_x = D_real.mean()
        C_x = C_real_loss

        D_loss = (D_real_loss + auto_D_fake_loss + ps_D_fake_loss + sep_D_fake_loss) + \
                    (C_real_loss + auto_C_fake_loss + ps_C_fake_loss + sep_C_fake_loss)/2
        D_loss.backward()
        self.optimizer_D.step()
        
        ############################################## G_loss ##############################################
        auto_D_fake, auto_C_fake = self.model.D(auto_G_out)
        REAL_LABEL = torch.ones_like(auto_D_fake)
        auto_D_fake_loss = self.bce_loss(auto_D_fake, REAL_LABEL)
        auto_C_fake_loss = self.cross_entropy_loss(auto_C_fake, labels)
        
        ps_D_fake, ps_C_fake = self.model.D(ps_G_out)
        REAL_LABEL = torch.ones_like(ps_D_fake)
        ps_D_fake_loss = self.bce_loss(ps_D_fake, REAL_LABEL)
        ps_C_fake_loss = self.cross_entropy_loss(ps_C_fake, labels)
        
        sep_D_fake, sep_C_fake = self.model.D(sep_G_out)
        REAL_LABEL = torch.ones_like(sep_D_fake)
        sep_D_fake_loss = self.bce_loss(sep_D_fake, REAL_LABEL)
        sep_C_fake_loss = self.cross_entropy_loss(sep_C_fake, labels)
                    
        auto_imgr_loss = self.l1_loss(auto_G_out, self.tanh(inputs))
        ps_imgr_loss = self.l1_loss(ps_G_out, self.tanh(inputs))
        sep_imgr_loss = self.l1_loss(sep_G_out, self.tanh(inputs))
        
        G_loss = (auto_D_fake_loss + ps_D_fake_loss + sep_D_fake_loss) + \
                    (auto_C_fake_loss + ps_C_fake_loss + sep_C_fake_loss)*2 + \
                    (auto_imgr_loss + ps_imgr_loss + sep_imgr_loss)*10
        ############################################################################################
        return D_loss, G_loss, auto_imgr_loss, ps_imgr_loss, sep_imgr_loss

    def forward(self, inputs, labels, batch):
        self.set_model()
        outputs = self.model.C(inputs)
                
        if opt.stage == 1:
            outputs_teacher = self.teacher(inputs)
            CrossEntropy_Loss = self.id_related_loss_1(labels, outputs, outputs_teacher)
            loss_sum = CrossEntropy_Loss
            
            print('\rCE:%.2f' % (CrossEntropy_Loss.data.cpu().numpy()), end=' ')
                    
        elif opt.stage == 2:
            D_loss, G_loss, auto_imgr_loss, ps_imgr_loss, sep_imgr_loss\
                    = self.GAN_loss(inputs, outputs, labels)
            KL_loss = self.KL_loss(outputs)
            
            loss_sum = G_loss + KL_loss/1000
                        
            print('\rD_loss:%.2f  G_loss:%.2f A_ImgR:%.2f  PS_ImgR:%.2f  Sep_PS:%.2f  KL:%.2f' % (
                D_loss.data.cpu().numpy(),
                G_loss.data.cpu().numpy(),
                auto_imgr_loss.data.cpu().numpy(),
                ps_imgr_loss.data.cpu().numpy(),
                sep_imgr_loss.data.cpu().numpy(),
                KL_loss.data.cpu().numpy()), end=' ')
                    
        elif opt.stage == 3:
            CrossEntropy_Loss = self.id_related_loss_3(labels, outputs)
            D_loss, G_loss, auto_imgr_loss, ps_imgr_loss, sep_imgr_loss\
                    = self.GAN_loss(inputs, outputs, labels)
            KL_loss = self.KL_loss(outputs)
                        
            loss_sum = (CrossEntropy_Loss*2)*10 + G_loss + KL_loss/100
            
            print('\rCE:%.2f  D_loss:%.2f  G_loss:%.2f  A_ImgR:%.2f  PS_ImgR:%.2f  Sep_PS:%.2f  KL:%.2f' % (
                CrossEntropy_Loss.data.cpu().numpy(),
                D_loss.data.cpu().numpy(),
                G_loss.data.cpu().numpy(),
                auto_imgr_loss.data.cpu().numpy(),
                ps_imgr_loss.data.cpu().numpy(),
                sep_imgr_loss.data.cpu().numpy(),
                KL_loss.data.cpu().numpy()), end=' ')
            
        return loss_sum


class CenterLoss(nn.Module):
    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    def forward(self, x, labels):
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)
    if return_inds:
        ind = (labels.new().resize_as_(labels)
                .copy_(torch.arange(0, N).long())
                .unsqueeze(0).expand(N, N))
        p_inds = torch.gather(
                ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
                ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds
    return dist_ap, dist_an

class TripletLoss(object):
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()
    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
