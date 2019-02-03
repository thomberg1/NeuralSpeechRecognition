import math
import statistics as stats

import torch
import torch.nn.functional as f


# from ctcdecode import CTCBeamDecoder

#######################################################################################################################

class STSDecoder(object):
    def __init__(self, vocab):
        self.vocab = vocab

    @staticmethod
    def decode_labels(labels, label_sizes, vocab):
        idx_sos, idx_eos, idx_pad = vocab('<SOS>'), vocab('<EOS>'), vocab('<PAD>')
        lseq = []
        for seq, size in zip(labels, label_sizes):
            lseq.append(
                ''.join([vocab(c.item()) for c in seq[0:size - 1] if c.item() not in [idx_sos, idx_eos, idx_pad]])
            )

        return lseq

    @staticmethod
    def decode_probas(probas, probas_sizes, vocab, probabilities=False):
        max_vals, max_indices = torch.max(probas, 2)
        idx_sos, idx_eos, idx_pad = vocab('<SOS>'), vocab('<EOS>'), vocab('<PAD>')

        decoded_seq = []
        for seq_idx, seq_len, seq_proba in zip(max_indices.cpu(), probas_sizes, max_vals):
            txt, probas = '', []

            for i in range(min(seq_len, len(seq_idx))):
                c = seq_idx[i].item()
                if c in [idx_sos, idx_eos, idx_pad]:
                    continue
                txt += vocab(c)
                probas.append(math.exp(seq_proba[i].item()))

            if probabilities:
                decoded_seq.append((txt.strip(), stats.mean(probas) if len(probas) > 0 else 0))
            else:
                decoded_seq.append(txt.strip())
        return decoded_seq

    def __call__(self, inputs, inputs_sizes, labels=None, label_sizes=None, probabilities=False):

        decoder_seq = self.decode_probas(inputs, inputs_sizes, self.vocab, probabilities=probabilities)

        label_seq = None
        if labels is not None and label_sizes is not None:
            label_seq = self.decode_labels(labels, label_sizes, self.vocab)

        return decoder_seq, label_seq


#######################################################################################################################


class CTCGreedyDecoder(object):
    def __init__(self, vocab):
        self.vocab = vocab

    @staticmethod
    def encode(labels, label_sizes):
        labels = [t[0][0:t[1]] for t in zip(labels, label_sizes)]
        labels = torch.cat(labels, 0).type(torch.IntTensor)
        return labels

    @staticmethod
    def decode_probas(inputs, input_sizes, vocab, remove_repetitions=False, filler='', probabilities=False):

        probas = f.softmax(inputs, dim=-1)
        max_vals, max_indices = torch.max(probas, 2)

        decoded_seq = []
        for seq_idx, seq_len, seq_proba in zip(max_indices.cpu(), input_sizes, max_vals):
            txt = ''
            probas = []
            for i in range(seq_len):
                if remove_repetitions and seq_idx[i] and i > 0 and seq_idx[i - 1] == seq_idx[i]:
                    continue
                c = seq_idx[i].item()
                txt += vocab(c) if c else filler
                if c:
                    probas.append(seq_proba[i].item())

            if probabilities:
                decoded_seq.append((txt, stats.mean(probas) if len(probas) > 0 else 0))
            else:
                decoded_seq.append(txt)

        return decoded_seq

    @staticmethod
    def decode_labels(inputs, input_sizes, vocab):

        decoded_seq = []
        for seq, seq_len in zip(inputs, input_sizes):
            txt = ''.join([vocab(seq[i].item()) for i in range(seq_len.item())])
            decoded_seq.append(txt)
        return decoded_seq

    def __call__(self, inputs, input_sizes, labels, label_sizes, probabilities=False):
        # logits : batchSize x seqLength x alphabet_size
        # logit_sizes : batchSize

        decoder_seq = self.decode_probas(inputs, input_sizes, self.vocab, remove_repetitions=True,
                                         filler='', probabilities=probabilities)

        label_seq = None
        if labels is not None:
            label_seq = self.decode_labels(labels, label_sizes, self.vocab)

        return decoder_seq, label_seq


#######################################################################################################################

# https://github.com/parlance/ctcdecode/blob/master/tests/test.py


class CTCBeamSearchDecoder(object):
    def __init__(self, vocabulary, lm_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=20,
                 num_processes=8, blank_index=0):
        self.vocab = vocabulary
        self.vocab_list = list(str(self.vocab))
        print(self.vocab_list)
        self.ctcdecoder = CTCBeamDecoder(self.vocab_list, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width,
                                         num_processes, blank_index)

    @staticmethod
    def encode(label_variables, label_size):
        labels = [t[0][0:t[1]] for t in zip(label_variables, label_size)]
        labels = torch.cat(labels, 0).type(torch.IntTensor)
        return labels

    def decode(self, input_vars, input_size):

        decoded_seq = []
        for seq, seq_len in zip(input_vars, input_size):
            txt = ''.join([self.vocab(x.item()) for x in seq[0:seq_len]])
            decoded_seq.append(txt)
        return decoded_seq

    def __call__(self, input_vars, label_vars, input_size, label_size, probabilities=False):
        # input_vars : batchSize x seqLength x alphabet_size
        # input_size : batchSize

        probs_seq = input_vars.cpu().data.contiguous()
        probs_size = input_size.cpu()

        out_seq, _, _, out_seq_len = self.ctcdecoder.decode(probs_seq, probs_size)

        decoder_seq = self.decode(out_seq[:, 0, :], out_seq_len[:, 0])

        label_seq = None
        if label_vars is not None:
            label_vars = label_vars.cpu()
            label_size = label_size.cpu()

            label_seq = self.decode(label_vars, label_size)

        return decoder_seq, label_seq

#######################################################################################################################
