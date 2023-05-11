import torch
from torch import Tensor
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np

from typing import Literal, Dict, Type, Union
from enum import Enum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NNType(Enum):
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"

    def __str__(self):
        return self.value


NN_MAP: Dict[NNType, Type[Union[nn.RNN, nn.LSTM, nn.GRU]]] = {
    NNType.RNN: nn.RNN,
    NNType.LSTM: nn.LSTM,
    NNType.GRU: nn.GRU,
}


class AttentionMethod(Enum):
    DOT_PRODUCT = "dot_product"
    SCALE_DOT_PRODUCT = "scale_dot_product"
    COSINE_SIMILARITY = "cosine_similarity"

    def __str__(self):
        return self.value


class DecoderBiRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        embedding: nn.Embedding,
        max_length: int,
        nn_type: Literal["rnn", "lstm", "gru"] = "rnn",
        num_layers=1,
        dropout_p=0.1,
        attention_method: Literal[
            "dot_product",
            "scale_dot_product",
            "cosine_similarity",
        ] = "dot_product",
    ):
        super(DecoderBiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = embedding
        self.dropout = nn.Dropout(self.dropout_p)
        self.nn_type = NNType(nn_type)
        self.attention_method = AttentionMethod(attention_method)
        self.nn = NN_MAP[self.nn_type](
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
        )
        self.out = nn.Linear(self.hidden_size * 4, self.output_size)

    def cal_attention(self, hidden: Tensor, encoder_hiddens: Tensor):
        if self.attention_method == AttentionMethod.DOT_PRODUCT:
            if self.nn_type == NNType.LSTM:  # For BiLSTM
                energy = torch.bmm(hidden[0], encoder_hiddens.T.repeat(2, 1, 1))
                attn_weights = F.softmax(energy, dim=-1)
                attn_output = torch.bmm(attn_weights, encoder_hiddens.repeat(2, 1, 1))
                concat_output = torch.cat(
                    (attn_output[0], hidden[0][0], attn_output[1], hidden[0][1]), 1
                )
            else:  # For BiRNN & BiGRU
                energy = torch.bmm(hidden, encoder_hiddens.T.repeat(2, 1, 1))
                attn_weights = F.softmax(energy, dim=-1)
                attn_output = torch.bmm(attn_weights, encoder_hiddens.repeat(2, 1, 1))
                concat_output = torch.cat(
                    (attn_output[0], hidden[0], attn_output[1], hidden[1]), 1
                )

        elif self.attention_method == AttentionMethod.COSINE_SIMILARITY:
            if self.nn_type == NNType.LSTM:  # For LSTM
                cosine_similarity = nn.CosineSimilarity(dim=-1)
                h_n, c_n = hidden
                # h_n_reshaped = h_n.mean(dim=0, keepdim=True)
                attn_weights_f = F.softmax(
                    cosine_similarity(h_n[0].unsqueeze(0), encoder_hiddens), dim=-1
                )
                attn_output_f = torch.bmm(
                    attn_weights_f.unsqueeze(0), encoder_hiddens.unsqueeze(0)
                )
                attn_weights_b = F.softmax(
                    cosine_similarity(h_n[1].unsqueeze(0), encoder_hiddens), dim=-1
                )
                attn_output_b = torch.bmm(
                    attn_weights_b.unsqueeze(0), encoder_hiddens.unsqueeze(0)
                )
                concat_output = torch.cat(
                    (
                        attn_output_f[0],
                        h_n[0],
                        attn_output_b[0],
                        h_n[1],
                    ),
                    1,
                )

            else:  # For RNN & GRU
                cosine_similarity = nn.CosineSimilarity(dim=-1)
                # hidden_reshaped = hidden.mean(dim=0, keepdim=True)
                # print(hidden_reshaped.shape)
                attn_weights_f = F.softmax(
                    cosine_similarity(hidden[0].unsqueeze(0), encoder_hiddens), dim=-1
                )
                attn_output_f = torch.bmm(
                    attn_weights_f.unsqueeze(0), encoder_hiddens.unsqueeze(0)
                )
                attn_weights_b = F.softmax(
                    cosine_similarity(hidden[1].unsqueeze(0), encoder_hiddens), dim=-1
                )
                attn_output_b = torch.bmm(
                    attn_weights_b.unsqueeze(0), encoder_hiddens.unsqueeze(0)
                )
                concat_output = torch.cat(
                    (
                        attn_output_f[0],
                        hidden[0],
                        attn_output_b[0],
                        hidden[1],
                    ),
                    1,
                )
        else:
            if self.nn_type == NNType.LSTM:  # For LSTM
                energy = torch.bmm(
                    hidden[0], encoder_hiddens.T.repeat(2, 1, 1)
                ) / np.sqrt(self.hidden_size)
                attn_weights = F.softmax(energy, dim=-1)
                attn_output = torch.bmm(attn_weights, encoder_hiddens.repeat(2, 1, 1))
                concat_output = torch.cat(
                    (attn_output[0], hidden[0][0], attn_output[1], hidden[0][1]), 1
                )
            else:  # For RNN & GRU
                energy = torch.bmm(hidden, encoder_hiddens.T.repeat(2, 1, 1)) / np.sqrt(
                    self.hidden_size
                )
                attn_weights = F.softmax(energy, dim=-1)
                attn_output = torch.bmm(attn_weights, encoder_hiddens.repeat(2, 1, 1))
                concat_output = torch.cat(
                    (attn_output[0], hidden[0], attn_output[1], hidden[1]), 1
                )
        return concat_output

    def forward(self, input: nn.Embedding, hidden: Tensor, encoder_hiddens: Tensor):
        embedded: Tensor = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        _, hidden = self.nn(embedded, hidden)
        concat_output = self.cal_attention(hidden, encoder_hiddens)
        output = F.log_softmax(self.out(concat_output), dim=1)
        return output, hidden

    def initHidden(self):
        return (
            torch.zeros(2, 1, self.hidden_size, device=device)
            if self.nn_type != NNType.LSTM
            else (
                torch.zeros(2, 1, self.hidden_size, device=device),
                torch.zeros(2, 1, self.hidden_size, device=device),
            )
        )
