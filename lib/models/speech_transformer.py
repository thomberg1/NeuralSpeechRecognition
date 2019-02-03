import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, droput, len_max=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.droput = droput
        self.len_max = len_max

        position = torch.arange(0.0, self.len_max)
        num_timescales = self.d_model // 2
        log_timescale_increment = math.log(10000) / (num_timescales - 1)
        inv_timescales = torch.exp(torch.arange(0.0, num_timescales) * -log_timescale_increment)
        scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
        pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)

        # wrap in a buffer so that model can be moved to GPU
        self.register_buffer('pos_emb', pos_emb)

        self.drop = nn.Dropout(self.droput)

    def forward(self, word_emb):
        len_seq = word_emb.size(1)
        out = word_emb + self.pos_emb[:len_seq, :]
        out = self.drop(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, droput):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.droput = droput

        self.d_head = d_model // self.num_heads

        self.fc_query = nn.Linear(self.d_model, self.num_heads * self.d_head, bias=False)
        self.fc_key = nn.Linear(self.d_model, self.num_heads * self.d_head, bias=False)
        self.fc_value = nn.Linear(self.d_model, self.num_heads * self.d_head, bias=False)

        self.fc_concat = nn.Linear(self.num_heads * self.d_head, self.d_model, bias=False)

        self.softmax = nn.Softmax(dim=1)

        self.attn_dropout = nn.Dropout(self.droput)
        self.dropout = nn.Dropout(self.droput)

        self.norm = nn.LayerNorm(self.d_model)

    def _prepare_proj(self, x):
        """Reshape the projectons to apply softmax on each head
        """
        b, l, d = x.size()
        return x.view(b, l, self.num_heads, self.d_head).transpose(1, 2).contiguous().view(b * self.num_heads, l,
                                                                                           self.d_head)

    def forward(self, query, key, value, mask):
        b, len_query = query.size(0), query.size(1)
        len_key = key.size(1)

        # project inputs to multi-heads
        proj_query = self.fc_query(query)  # batch_size x len_query x h*d_head
        proj_key = self.fc_key(key)  # batch_size x len_key x h*d_head
        proj_value = self.fc_value(value)  # batch_size x len_key x h*d_head

        # prepare the shape for applying softmax
        proj_query = self._prepare_proj(proj_query)  # batch_size*h x len_query x d_head
        proj_key = self._prepare_proj(proj_key)  # batch_size*h x len_key x d_head
        proj_value = self._prepare_proj(proj_value)  # batch_size*h x len_key x d_head

        # get dotproduct softmax attns for each head
        attns = torch.bmm(proj_query, proj_key.transpose(1, 2))  # batch_size*h x len_query x len_key
        attns = attns / math.sqrt(self.d_head)
        attns = attns.view(b, self.num_heads, len_query, len_key)
        attns = attns.masked_fill_(mask.unsqueeze(1), -float('inf'))
        attns = self.softmax(attns.view(-1, len_key))

        # return mean attention from all heads as coverage
        coverage = torch.mean(attns.view(b, self.num_heads, len_query, len_key), dim=1)

        attns = self.attn_dropout(attns)
        attns = attns.view(b * self.num_heads, len_query, len_key)

        # apply attns on value
        out = torch.bmm(attns, proj_value)  # batch_size*h x len_query x d_head
        out = out.view(b, self.num_heads, len_query, self.d_head).transpose(1, 2).contiguous()

        out = self.fc_concat(out.view(b, len_query, self.num_heads * self.d_head))

        out = self.dropout(out).add_(query)
        out = self.norm(out)
        return out, coverage


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        out = self.fc(inputs)
        out = self.drop(out).add_(inputs)
        out = self.norm(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, dropout, d_ff):
        super(EncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.attention = MultiHeadAttention(self.num_heads, self.d_model, self.dropout)

        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)

    def forward(self, query, key, value, mask):
        out, _ = self.attention(query, key, value, mask)
        out = self.ff(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, dropout, d_ff):
        super(DecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.attention_tgt = MultiHeadAttention(self.num_heads, self.d_model, self.dropout)

        self.attention_src = MultiHeadAttention(self.num_heads, self.d_model, self.dropout)

        self.ff = PositionwiseFeedForward(d_model, self.d_ff, self.dropout)

    def forward(self, query, key, value, context, mask_tgt, mask_src):
        out, _ = self.attention_tgt(query, key, value, mask_tgt)
        out, coverage = self.attention_src(out, context, context, mask_src)
        out = self.ff(out)
        return out, coverage


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_heads, d_model, dropout, d_ff, num_layers=6, padding_idx=1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout

        self.embeddings = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.padding_idx)

        self.pos_emb = PositionalEncoding(self.d_model, self.dropout, len_max=512)

        self.layers = nn.ModuleList(
            [EncoderLayer(self.num_heads, self.d_model, self.dropout, self.d_ff) for _ in range(self.num_layers)]
        )

    def forward(self, src):
        context = self.embeddings(src)  # batch_size x len_src x d_model

        context = self.pos_emb(context)

        mask_src = src.data.eq(self.padding_idx).unsqueeze(1)
        for _, layer in enumerate(self.layers):
            context = layer(context, context, context, mask_src)  # batch_size x len_src x d_model
        return context, mask_src


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_heads, d_model, dropout, d_ff, num_layers=6, padding_idx=1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.padding_idx)

        self.pos_emb = PositionalEncoding(self.d_model, self.dropout, len_max=512)

        self.layers = nn.ModuleList(
            [DecoderLayer(self.num_heads, self.d_model, self.dropout, self.d_ff) for _ in range(self.num_layers)]
        )

        self.fc = nn.Linear(self.d_model, self.vocab_size, bias=True)

        # tie weight between word embedding and generator
        self.fc.weight = self.embedding.weight

        self.logsoftmax = nn.LogSoftmax(dim=1)

        # pre-save a mask to avoid future information in self-attentions in decoder
        # save as a buffer, otherwise will need to recreate it and move to GPU during every call
        mask = torch.ByteTensor(np.triu(np.ones((self.d_model, self.d_model)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

    def forward(self, tgt, context, mask_src):
        out = self.embedding(tgt)  # batch_size x len_tgt x d_model

        out = self.pos_emb(out)

        len_tgt = tgt.size(1)
        mask_tgt = tgt.data.eq(self.padding_idx).unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)
        for _, layer in enumerate(self.layers):
            out, coverage = layer(out, out, out, context, mask_tgt, mask_src)  # batch_size x len_tgt x d_model

        out = self.fc(out)  # batch_size x len_tgt x bpe_size

        out = self.logsoftmax(out.view(-1, self.vocab_size))
        return out, coverage


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, num_heads, d_model, dropout, d_ff, num_layers=6, padding_idx=1):
        super(Transformer, self).__init__()
        self.src_vocab = src_vocab
        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab = tgt_vocab
        self.tgt_vocab_size = len(tgt_vocab)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.padding_idx = padding_idx

        self.encode = Encoder(self.src_vocab_size, self.num_heads, self.d_model, self.dropout, self.d_ff,
                              self.num_layers, self.padding_idx)
        self.decode = Decoder(self.tgt_vocab_size, self.num_heads, self.d_model, self.dropout, self.d_ff,
                              self.num_layers, self.padding_idx)

    def forward(self, src, tgt):
        context, mask_src = self.encode(src)
        outputs, _ = self.decode(tgt, context, mask_src)

        probas = outputs.view(src.size(0), -1, self.tgt_vocab_size)
        _, max_indices = torch.max(probas, 2)
        proba_sizes = torch.max(max_indices.eq(self.tgt_vocab('<eos>')), dim=1)[1] + 1

        return probas, proba_sizes

    def decode_greedy(self, inputs, labels=None, max_seq_length=50):

        idx_sos, idx_eos = self.tgt_vocab('<sos>'), self.tgt_vocab('<eos>')

        context, mask_src = self.encode(inputs)

        batch_size = inputs.size(0)
        decode_input = torch.ones(batch_size, 1).fill_(idx_sos).type_as(inputs)

        dec_output_sizes = torch.LongTensor(batch_size).fill_(max_seq_length).type_as(inputs)

        max_steps = labels.size(1) if labels is not None else max_seq_length + 1

        dec_outputs = []
        for step in range(max_steps):
            outputs, _ = self.decode(decode_input, context, mask_src)
            outputs = outputs.view(batch_size, -1, self.tgt_vocab_size)

            dec_outputs.append(outputs[:, step, :].unsqueeze(1))

            preds = torch.max(outputs[:, -1, :], dim=1)[1]

            dec_output_sizes[preds.eq(idx_eos) * dec_output_sizes.gt(step)] = step
            if labels is None and dec_output_sizes.le(step + 1).all():
                break

            decode_input = torch.cat([decode_input, preds.unsqueeze(1)], dim=1)

        dec_outputs = torch.cat(dec_outputs, dim=1)

        return dec_outputs, dec_output_sizes

    def decode_beam(self, inputs, labels=None, max_seq_length=50, beam_size=64, alpha=0.1, beta=0.3):

        context, mask_src = self.encode(inputs)

        max_seq_len = labels.size(1) if labels is not None else max_seq_length

        dec_outputs = []
        for idx in range(context.size(0)):
            target, _ = beam_search(self, self.tgt_vocab, context[idx].unsqueeze(0), mask_src[idx].unsqueeze(0),
                                    beam_size=beam_size, alpha=alpha, beta=beta, max_seq_len=max_seq_len)
            dec_outputs.append(target)

        return dec_outputs


def beam_search(model, vocab, context, mask_src, beam_size=64, alpha=0.1, beta=0.3, max_seq_len=64):
    probas = []
    preds = []
    probs = []
    coverage_penalties = []

    vocab_size = len(vocab)
    idx_sos, idx_eos, idx_pad = vocab('<sos>'), vocab('<eos>'), vocab('<pad>')

    decode_inputs = torch.LongTensor([idx_sos]).unsqueeze(1)
    if next(model.parameters()).is_cuda:
        decode_inputs = decode_inputs.cuda()

    decode_outputs, coverage = model.decode(decode_inputs, context, mask_src)

    scores, scores_idx = decode_outputs.view(-1).topk(beam_size)
    beam_idx = scores_idx / vocab_size
    pred_idx = (scores_idx - beam_idx * vocab_size).view(beam_size, -1)

    decode_inputs = torch.cat((decode_inputs.repeat(beam_size, 1), pred_idx), 1)
    context = context.repeat(beam_size, 1, 1)

    remaining_beams = beam_size
    for step in range(max_seq_len):
        decode_outputs, coverage = model.decode(decode_inputs, context, mask_src)

        decode_outputs = decode_outputs.view(remaining_beams, -1, vocab_size)
        decode_outputs = scores.unsqueeze(1) + decode_outputs[:, -1, :]
        scores, scores_idx = decode_outputs.view(-1).topk(remaining_beams)

        beam_idx = scores_idx / vocab_size
        pred_idx = (scores_idx - beam_idx * vocab_size).view(remaining_beams, -1)

        decode_inputs = torch.cat((decode_inputs[beam_idx], pred_idx), 1)

        index = decode_inputs[:, -1].eq(idx_eos) + decode_inputs[:, -1].eq(idx_pad)
        finished = index.nonzero().flatten()
        continue_idx = (index ^ 1).nonzero().flatten()

        for idx in finished:
            probas.append(scores[idx].item())
            preds.append(decode_inputs[idx, :].tolist())
            probs.append(coverage[idx, :, :])

            atten_prob = torch.sum(coverage[idx, :, :], dim=0)
            coverage_penalty = torch.log(atten_prob.masked_select(atten_prob.le(1)))
            coverage_penalty = beta * torch.sum(coverage_penalty).item()
            coverage_penalties.append(coverage_penalty)

            remaining_beams -= 1

        if len(continue_idx) > 0:
            scores = scores.index_select(0, continue_idx)
            decode_inputs = decode_inputs.index_select(0, continue_idx)
            context = context.index_select(0, continue_idx)

        if remaining_beams <= 0:
            break

    len_penalties = [math.pow(len(pred), alpha) for pred in preds]
    #     final_scores = [probas[i] / len_penalties[i] + coverage_penalties[i] for i in range(len(preds))]
    final_scores = [probas[i] / len_penalties[i] for i in range(len(preds))]

    sorted_scores_arg = sorted(range(len(preds)), key=lambda i: -final_scores[i])

    best_beam = sorted_scores_arg[0]

    return preds[best_beam], probs[best_beam]
