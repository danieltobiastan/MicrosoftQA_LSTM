import torch
from torch import Tensor
import torch.nn as nn
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


class DocumentBiRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding: nn.Embedding,
        nn_type: Literal["rnn", "lstm", "gru"] = "rnn",
        num_layers=1,
    ):
        super(DocumentBiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.nn_type = NNType(nn_type)
        self.nn = NN_MAP[self.nn_type](
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
        )

    def forward(self, input: nn.Embedding, hidden: Tensor):
        embedded: Tensor = self.embedding(input).view(1, 1, -1)
        output: Tensor
        output, hidden = self.nn(embedded, hidden)
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
