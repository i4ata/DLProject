"""
Transformer model undone scratch (gotta finish AFT)
To do and test:
1) Finish AFT_Local/Simple
2) Finish AFTransformer
3) Add gpu support
4) Implement training routine
"""

from aft import AFTLocal
from typing import Tuple

import torch
import torch.nn as nn


class FF(nn.Module):
    def __init__(self, e_dim: int, hid_dim: int, dropout_rate: float = 0.1):
        super(FF, self).__init__()
        self.ff = nn.Sequential(  # the original paper used FFs of this sort
            nn.Linear(e_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, e_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.ff(x)

class DecoderBlock(nn.Module):
    def __init__(self,
                 e_dim: int,
                 hid_dim: int,
                 dropout_rates: Tuple[float, float] = (0.1, 0.1),
                 attention: bool = True):
        super().__init__()
        self.attention = AFTLocal(...)  # gotta finish this one

        self.dropout_1 = nn.Dropout(dropout_rates[0])
        self.norm = nn.LayerNorm(e_dim)

        self.ff = FF(e_dim, hid_dim)
        self.dropout_2 = nn.Dropout(dropout_rates[1])

    def forward(self, x: torch.Tensor, pos_encod: torch.Tensor):
        
        embedding = x + pos_encod

        # attention -> dropout -> residual connection -> layer normalization
        attn_out = self.norm(self.dropout_1(self.atf(embedding, embedding, embedding)) + embedding)

        # mlp -> dropout -> residual connection -> layer normalization
        ff_out = self.norm(self.dropout_1(self.ff(attn_out)) + attn_out)

        return ff_out

class DecoderStack(nn.Module):
    def __init__(self,
                 layers: int,
                 e_dim: int,
                 hid_dim: int,
                 vocab_size: int,
                 max_sequence_len: int,
                 dropout_rates: Tuple[float] = (0.1, 0.1)):
        super(DecoderStack, self).__init__()
        self.layers = layers
        self.e_dim = e_dim
        self.hid_dim = hid_dim
        self.max_sequence_len = max_sequence_len
        self.vocab_size = vocab_size
        self.scale_factor = torch.rsqrt(torch.as_tensor(e_dim, dtype=torch.float32))

        self.token_embedding = nn.Embedding(vocab_size, e_dim)
        self.position_embedding = nn.Embedding(max_sequence_len, e_dim)
        self.embedding_dropout = nn.Dropout(dropout_rates[0])

        self.decoder_layers = nn.ModuleList([  # makes the stack of decoder layers
            DecoderBlock(e_dim, hid_dim, (dropout_rates[0], dropout_rates[1]))
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        sequence_len = x.shape[1]
        position_indices = torch.arange(sequence_len).expand((x.shape[0], -1))

        token_embeddings = self.token_embedding(x) * self.scale_factor
        position_embeddings = self.position_embedding(position_indices) * self.scale_factor

        decoder_output = self.embedding_dropout(token_embeddings) + self.embedding_dropout(position_embeddings)

        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, position_embeddings)

        return decoder_output


class DecoderOnlyAFT(nn.Module):
    def __init__(self,
                 layers: int,
                 e_dim: int,
                 hid_dim: int,
                 vocab_size: int,
                 max_sequence_len: int,
                 dropout_rates: Tuple[float] = (0.1, 0.1)):
        super(DecoderOnlyAFT, self).__init__()

        self.max_sequence_len = max_sequence_len
        self.vocab_size = vocab_size

        self.loss_fn = nn.CrossEntropyLoss(reduction="none") # corss-entropy, for now without summing or averaging over the batch

        self.decoder_stack = DecoderStack(layers=layers,
                                          e_dim=e_dim,
                                          hid_dim=hid_dim,
                                          vocab_size=vocab_size,
                                          max_sequence_len=max_sequence_len,
                                          dropout_rates=dropout_rates)

        self.project_to_vocab = nn.Linear(e_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        dec_outputs = self.decoder_stack(x)
        return dec_outputs

    def compute_loss(self, x_in: torch.Tensor, x_out: torch.Tensor, seg_len=None):
        pass

    def infer(self, x: torch.Tensor):
        pass