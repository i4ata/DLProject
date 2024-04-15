"""
Transformer model undone scratch (gotta finish AFT)
To do and test:
1) Finish AFT_Local/Simple
2) Finish AFTransformer
3) Add gpu support
4) Implement training routine
"""

from aft import AFTLocal, AFTSimple
from typing import Tuple, Literal

import torch
import torch.nn as nn
import math

BEST_WINDOW_SIZE = 32 # according to the paper

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
                 sequence_len: int,
                 dropout_rates: Tuple[float, float] = (0.1, 0.1),
                 aft: Literal['simple', 'local'] = 'simple'):
        super().__init__()
        # self.aft = AFTLocal(e_dim=e_dim, sequence_len=sequence_len, s=BEST_WINDOW_SIZE)
        if aft == 'simple':
            self.aft = AFTSimple(e_dim=e_dim, sequence_len=sequence_len)
        else:
            self.aft = AFTLocal(e_dim=e_dim, sequence_len=sequence_len, s=BEST_WINDOW_SIZE)


        self.dropout_1 = nn.Dropout(dropout_rates[0])
        self.norm = nn.LayerNorm(e_dim)

        self.ff = FF(e_dim, hid_dim)
        self.dropout_2 = nn.Dropout(dropout_rates[1])

    def forward(self, x: torch.Tensor, pos_encod: torch.Tensor):
        
        embedding = x + pos_encod

        # attention -> dropout -> residual connection -> layer normalization
        attn_out = self.norm(self.dropout_1(self.aft(embedding, embedding, embedding)) + embedding)

        # mlp -> dropout -> residual connection -> layer normalization
        ff_out = self.norm(self.dropout_1(self.ff(attn_out)) + attn_out)

        return ff_out

class DecoderStack(nn.Module):
    def __init__(self,
                 layers: int,
                 e_dim: int,
                 hid_dim: int,
                 vocab_size: int,
                 sequence_len: int,
                 dropout_rates: Tuple[float] = (0.1, 0.1),
                 aft: Literal['simple', 'local'] = 'simple'):
        super(DecoderStack, self).__init__()
        self.layers = layers
        self.e_dim = e_dim
        self.hid_dim = hid_dim
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.scale_factor = 1 / math.sqrt(e_dim)

        self.token_embedding = nn.Embedding(vocab_size, e_dim)
        self.position_embedding = nn.Embedding(sequence_len, e_dim)
        self.embedding_dropout = nn.Dropout(dropout_rates[0])

        self.decoder_layers = nn.ModuleList([  # makes the stack of decoder layers
            DecoderBlock(e_dim=e_dim, hid_dim=hid_dim, sequence_len=sequence_len, dropout_rates=(dropout_rates[0], dropout_rates[1]), aft=aft)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        position_indices = torch.arange(self.sequence_len).expand((x.shape[0], -1)).to(x.device)

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
                 sequence_len: int,
                 dropout_rates: Tuple[float] = (0.1, 0.1),
                 aft: Literal['simple', 'local'] = 'simple'):
        super(DecoderOnlyAFT, self).__init__()

        self.sequence_len = sequence_len
        self.vocab_size = vocab_size

        #self.loss_fn = nn.CrossEntropyLoss(reduction="none") # corss-entropy, for now without summing or averaging over the batch
        
        self.decoder_stack = DecoderStack(layers=layers,
                                          e_dim=e_dim,
                                          hid_dim=hid_dim,
                                          vocab_size=vocab_size,
                                          sequence_len=sequence_len,
                                          dropout_rates=dropout_rates,
                                          aft=aft)

        self.project_to_vocab = nn.Linear(e_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        dec_outputs = self.decoder_stack(x)
        projected_to_vocab = self.project_to_vocab(dec_outputs[:, -1]) # Added that, might be wrong
        return projected_to_vocab
    
    # inference function unused and untested properly
    # def infer(self, x: torch.Tensor, candidates_to_consider=1):
    #     infer_seqs = [x[:, :1]]  # start with the first token in every sequence of the batch

    #     for step in range(1, self.sequence_len):
    #         tmp_inputs = torch.cat(infer_seqs, dim=1) # concatenate the tokens from the previous steps
            
    #         with torch.no_grad():
    #             dec_out = self.forward(tmp_inputs, training=False)
    #             logits = self.project_to_vocab(dec_out[:, step:step+1, :])

    #         if candidates_to_consider == 1:
    #             temp_argmax = logits.argmax(dim=-1) # greedy decoding
    #         else:
    #             temp_prob = nn.functional.softmax(logits, dim=-1) 
    #             temp_top_k = torch.topk(temp_prob, k=candidates_to_consider, dim=-1) # get the top k candidates for next token
    #             temp_sample = torch.multinomial(temp_top_k.values.squeeze(-1), 1) # sample from the top k candidates
    #             temp_argmax = torch.gather(temp_top_k.indices, -1, temp_sample) 

    #         infer_seqs.append(temp_argmax) # append the next tokens to their corresponding sequences

    #     return torch.cat(infer_seqs, dim=1)
