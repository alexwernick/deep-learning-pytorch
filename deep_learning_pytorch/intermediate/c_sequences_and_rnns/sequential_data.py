'''
Sequential data
- Ordered in time or space
- Order of the data points contains dependencies between them
- Examples of sequential data
  - Time series
  - Text
  - Audio Waves

  Data has been taken from https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

  Train-test split
  - No random splitting for time series!
  - Look-ahead bias: model has info about the future
  - Solution is to split the data by time

Creating Sequences
- Sequence length = number of data points in one training example
  - 24 x 4 = 96 -> consider last 24 hours
- Predict single next data point

'''

import numpy as np

def create_sequences(df, seq_length):
    xs, ys = [], []
    for i in range(len(df) - seq_length):
        x = df.iloc[i:(i+seq_length), 1]
        y = df.iloc[i+seq_length, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)