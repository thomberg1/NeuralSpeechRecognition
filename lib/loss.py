import torch
import torch.nn as nn
import torch.nn.functional as f


#######################################################################################################################

class PytorchCTCLoss(object):

    def __init__(self, cuda=True):
        self.cuda = cuda
        self.loss = nn.CTCLoss(blank=0, reduction='sum')

    def __call__(self, outputs, output_sizes, labels, label_sizes):
        return self.forward(outputs, output_sizes, labels, label_sizes)

    def forward(self, logits, logit_sizes, labels, label_sizes):
        log_probs = f.log_softmax(logits, dim=-1)

        if not self.cuda:  # hack to push pytorch to execute CTCloss on the cpu even if model is on gpu
            log_probs = log_probs.cpu()
            labels = labels.cpu()

        loss = self.loss(log_probs.transpose(1, 0), labels, logit_sizes, label_sizes)

        return loss


#######################################################################################################################

class STSNLLLoss(object):
    def __init__(self, padding_idx=0):
        self.padding_idx = padding_idx

        self.loss = nn.NLLLoss(ignore_index=self.padding_idx, reduction='sum')

    def __call__(self, outputs, output_sizes, labels, label_sizes):
        return self.forward(outputs, output_sizes, labels, label_sizes)

    def forward(self, outputs, output_sizes, labels, label_sizes):
        loss = self.loss(outputs.transpose(1, 2), labels)
        return loss


#######################################################################################################################

class LabelSmoothingLoss(nn.Module):
    def __init__(self, padding_idx, label_smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing

    def __call__(self, inputs, input_sizes, labels, label_sizes):
        return self.forward(inputs, input_sizes, labels, label_sizes)

    def forward(self, inputs, input_sizes, labels, label_sizes):
        b, t, c = inputs.size()
        inputs = inputs.view(b * t, c)

        b, t = labels.size()
        labels = labels.view(b * t)

        true_dist = inputs.clone()
        true_dist.fill_(self.smoothing / (inputs.size(1) - 2))
        true_dist.scatter_(1, labels.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0

        mask = torch.nonzero(labels == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return self.criterion(inputs, true_dist.detach())

#######################################################################################################################
