import torch
import torch.nn.functional as F
import torch.nn as nn

class partial_loss(nn.Module):
    def __init__(self, train_givenY):
        super().__init__()
        print('Calculating uniform targets...')
        tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(1, train_givenY.shape[1])
        confidence = train_givenY.float()/tempY
        confidence = confidence.cuda()
        # calculate confidence
        self.confidence = confidence

    def forward(self, outputs, index, targets=None):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        if targets is None:
            # using confidence
            final_outputs = logsm_outputs * self.confidence[index, :].detach()
        else:
            # using given tagets
            final_outputs = logsm_outputs * targets.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        average_loss = loss_vec.mean()
        return average_loss, loss_vec

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index):
        self.confidence[batch_index, :] = temp_un_conf
        return None

def consistency_loss(logits_w, logits_s, sin_label_idx, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        pred_w = torch.softmax(logits_w, dim=1).detach()
        pred_s = torch.softmax(logits_s, dim=1).detach()
        return F.mse_loss(pred_s, pred_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs = pseudo_label[range(pseudo_label.shape[0]), sin_label_idx]
        mask = max_probs.ge(p_cutoff).float()
        
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, sin_label_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')