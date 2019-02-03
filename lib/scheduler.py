import torch


#######################################################################################################################

class NoamLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, warmup_steps, factor=1.0, last_epoch=-1):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.factor = factor
        super(NoamLRScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        rate = self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        return [rate for _ in range(len(self.base_lrs))]

#######################################################################################################################
