import torch
import torch.nn as nn


class DCL_Loss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, T=0.1):
        super(DCL_Loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.T = T

    def cal_intra(self, t1c_feat, t2h_feat):
        batchSize = t1c_feat.shape[0]
        pos = (t1c_feat * t2h_feat.data).sum(1).div_(self.T).exp_()  # exp(sum/T)  # torch.Size([16])
        pos_all_prob = torch.mm(t1c_feat, t2h_feat.t().data).div_(self.T).exp_()  # ai对每一个vk求和（余弦距离）
        pos_all_div = pos_all_prob.sum(1)
        pos_prob = torch.div(pos, pos_all_div)
        loss_pos = pos_prob.log_()
        lnpossum = loss_pos.sum(0)
        intra_loss = - lnpossum / batchSize
        return intra_loss

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # 计算intra-case 损失
        # 两两之间求相似
        t1c_feat = features[:, 0, :]
        t2h_feat = features[:, 1, :]
        intra_loss = self.cal_intra(t1c_feat, t2h_feat) / 2


       # 计算inter-case 损失
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.t()).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) #[batch, view, 128] - > [batch*view, 128]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        # print('anchor_feature', anchor_feature.shape)
        # print('contrast_feature', contrast_feature.shape)

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.t()),
            self.temperature)
        # print('anchor_dot_contrast', anchor_dot_contrast)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # 最大的数值
        # print('logits_max', logits_max, logits_max.shape)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # print('mask', mask)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  #除了自己都要比
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # 每一个元素都减去这一行的相似度的和
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        inter_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        inter_loss = inter_loss.view(anchor_count, batch_size).mean()

        loss = intra_loss + inter_loss


        return loss

if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    random_tensor = torch.randn(32, 2, 128)
    random_tensor = F.normalize(random_tensor, dim=2)
    label = torch.randint(0, 2, (32,))
    loss = DCL_Loss()(random_tensor, label)

