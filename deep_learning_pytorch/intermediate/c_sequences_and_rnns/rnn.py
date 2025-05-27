'''
- rnns are like feed forward neural networks but have connections pointing back
- Recurrent neuron:
  - Input X
  - Output y
  - Hidden state h
- In Pytorch: nn.RNN()

4 different types of RNNs
- Sequence to sequence
  - pass sequence as input and use the entire output sequence
  - e.g. real time speech recognition
- Sequence to vector architecture
  - Pass sequence as input, only use the last output
  - e.g. text topic classification
- Vector to sequence architecture
  - Pass single input, use entire output as sequence
  - e.g. text generation
- Encoder-decoder architecture
  - Pass entire input sequence, only then start using output sequence
  - e.g. machine translation

'''

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init_()
        self.rnn = nn.RNN(
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
        out, _ = self.rnn(x, h0)
        # select last RNN's output and pass through linear layer
        out = self.fc(out[:, -1, :])
        return out
