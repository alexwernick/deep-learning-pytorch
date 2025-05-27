'''
Short-term memory problem
- rnn cells maintain memory via hidden state
- the memory is very short term
- two more powerful cells have been proposed
  - LSTM (Long Short-Term Memory) cell
  - GRU (Gated Recurrent Unit) cell

RNN cell
- Two inputs
  - current input data x
  - previous hidden state h
- Two outputs
  - current output y
  - next hidden state h

LSTM cell
- Three inputs and outputs (two hidden states)
  - h: short term state
  - c: long term state
- Three 'gates'
  - Forget gate: what to remove from long term memory
  - Input gate: what to save to long term memory
  - Output gate: what to return at the current time step


GRU cell
- Simplified version of LSTM cell
- Just one hidden state
- No output gate


Should I use RNN, LSTM or GRU
- RNN is not used much anymore due to short term memory issue
- GRU is simpler than LSTM = less computation
- Relative performance varies per use-case
- Try both and compare
'''
import torch
import torch.nn as nn

class LSTMNet(nn.module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            # electricity value input
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
        # output vector of size 1
        # electricity value
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # 2 is the number of layers
        # x.size(0) is the input size
        # 32 is hidden state size
        # in effect we initialize first hidden state to zeros
        h0 = torch.zeros(2, x.size(0), 32)
        c0 = torch.zeros(2, x.size(0), 32)

        # pass input and both hidden states through lstm layer
        out, _ = self.lstm(x, (h0, c0))
        # select last RNN's output and pass through linear layer
        out = self.fc(out[:, -1, :])
        return out
    

class GRUNet(nn.Module):
    def __init__(self):
        super().__init_()
        self.gru = nn.GRU(
            # electricity value input
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
        # output vector of size 1
        # electricity value
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # 2 is the number of layers
        # x.size(0) is the input size
        # 32 is hidden state size
        # in effect we initialize first hidden state to zeros
        h0 = torch.zeros(2, x.size(0), 32)

        # pass input and hidden state through RNN layer
        out, _ = self.gru(x, h0)
        # select last RNN's output and pass through linear layer
        out = self.fc(out[:, -1, :])
        return out