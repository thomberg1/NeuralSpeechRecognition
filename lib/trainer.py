import torch


#######################################################################################################################


class Trainer(object):
    def __init__(self, model, loader, optimizer, scheduler, criterion, decoder, scorer, max_grade_norm=None):
        self.model = model
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.decoder = decoder
        self.scorer = scorer
        self.max_grade_norm = max_grade_norm
        self.use_cuda = next(self.model.parameters()).is_cuda

    def __call__(self, epoch):
        self.model.train(True)

        self.scheduler.step(epoch)

        train_lr = [float(param_group['lr']) for param_group in self.optimizer.param_groups][0]

        total_size, total_loss, total_score = 0, 0.0, 0.0
        for inputs, labels, input_sizes, label_sizes, _ in self.loader:
            if next(self.model.parameters()).is_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs, output_sizes = self.model(inputs, input_sizes, labels, label_sizes)

            loss = self.criterion(outputs, output_sizes, labels, label_sizes)
            total_loss += loss.item()

            preds_seq, label_seq = self.decoder(outputs, output_sizes, labels, label_sizes)
            total_score += self.scorer(preds_seq, label_seq)

            total_size += inputs.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grade_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grade_norm)
            self.optimizer.step()

            del outputs
            del loss

        return total_loss / total_size, 1.0 - min(1.0, total_score / total_size), train_lr

#######################################################################################################################
