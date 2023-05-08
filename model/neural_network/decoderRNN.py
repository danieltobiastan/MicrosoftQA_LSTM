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


class DecoderRNN(nn.Module):
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
        super(DecoderRNN, self).__init__()
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
        )
        self.attn = nn.Linear(self.hidden_size * 2, 1)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def cal_attention(self, hidden: Tensor, encoder_hiddens: Tensor):
        if self.attention_method == AttentionMethod.DOT_PRODUCT:
            if self.nn_type == NNType.LSTM:  # For LSTM
                attn_weights = F.softmax(
                    torch.bmm(hidden[0], encoder_hiddens.T.unsqueeze(0)), dim=-1
                )
                attn_output = torch.bmm(attn_weights, encoder_hiddens.unsqueeze(0))
                concat_output = torch.cat((attn_output[0], hidden[0][0]), 1)
            else:  # For RNN & GRU
                attn_weights = F.softmax(
                    torch.bmm(hidden, encoder_hiddens.T.unsqueeze(0)), dim=-1
                )
                attn_output = torch.bmm(attn_weights, encoder_hiddens.unsqueeze(0))
                concat_output = torch.cat((attn_output[0], hidden[0]), 1)

        elif self.attention_method == AttentionMethod.COSINE_SIMILARITY:
            if self.nn_type == NNType.LSTM:  # For LSTM
                cosine_similarity = nn.CosineSimilarity(dim=-1)
                attn_weights = F.softmax(
                    cosine_similarity(hidden[0], encoder_hiddens), dim=-1
                )
                attn_output = torch.bmm(
                    attn_weights.unsqueeze(0), encoder_hiddens.unsqueeze(0)
                )
                concat_output = torch.cat((attn_output[0], hidden[0][0]), 1)
            else:  # For RNN & GRU
                cosine_similarity = nn.CosineSimilarity(dim=-1)
                attn_weights = F.softmax(
                    cosine_similarity(hidden, encoder_hiddens), dim=-1
                )
                attn_output = torch.bmm(
                    attn_weights.unsqueeze(0), encoder_hiddens.unsqueeze(0)
                )
                concat_output = torch.cat((attn_output[0], hidden[0]), 1)

        # For scale dot product
        elif self.nn_type == NNType.LSTM:  # For LSTM
            energy = torch.bmm(hidden[0], encoder_hiddens.T.unsqueeze(0)) / np.sqrt(
                self.hidden_size
            )
            attn_weights = F.softmax(energy, dim=-1)
            attn_output = torch.bmm(attn_weights, encoder_hiddens.unsqueeze(0))
            concat_output = torch.cat((attn_output[0], hidden[0][0]), 1)

        else:  # For RNN & GRU
            attn_weights = F.softmax(
                torch.bmm(hidden, encoder_hiddens.T.unsqueeze(0))
                / np.sqrt(self.hidden_size),
                dim=-1,
            )
            attn_output = torch.bmm(attn_weights, encoder_hiddens.unsqueeze(0))
            concat_output = torch.cat((attn_output[0], hidden[0]), 1)
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
            torch.zeros(1, 1, self.hidden_size, device=device)
            if self.nn_type != NNType.LSTM
            else (
                torch.zeros(1, 1, self.hidden_size, device=device),
                torch.zeros(1, 1, self.hidden_size, device=device),
            )
        )
