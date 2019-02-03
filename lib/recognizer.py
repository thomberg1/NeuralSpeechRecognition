import torch


#######################################################################################################################


class Recognizer(object):
    def __init__(self, model, decoder, loader, probabilities=False):
        self.model = model
        self.decoder = decoder
        self.loader = loader
        self.probabilities = probabilities

        self.use_cuda = next(self.model.parameters()).is_cuda

    def __call__(self):

        self.model.eval()
        with torch.no_grad():

            decoder_seq = []
            for batch_idx, data in enumerate(self.loader):
                if len(data) == 3:
                    inputs, input_sizes, idx = data
                elif len(data) == 5:
                    inputs, _, input_sizes, _, _ = data
                else:
                    raise RuntimeError('Loader with invalid parameter count')

                if next(self.model.parameters()).is_cuda:
                    inputs = inputs.cuda()

                logits, logit_sizes = self.model(inputs, input_sizes)

                seq, _ = self.decoder(logits, logit_sizes, None, None, probabilities=self.probabilities)

                decoder_seq.extend(seq)

                del logits

        return decoder_seq

#######################################################################################################################
