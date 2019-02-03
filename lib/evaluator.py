import torch


#######################################################################################################################


class Evaluator(object):
    def __init__(self, model, loader, criterion, decoder, scorer):
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.decoder = decoder
        self.scorer = scorer
        self.use_cuda = next(self.model.parameters()).is_cuda

    def __call__(self):
        self.model.eval()

        with torch.no_grad():
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

                del outputs
                del loss

            return total_loss / total_size, 1.0 - min(1.0, total_score / total_size)

#######################################################################################################################
