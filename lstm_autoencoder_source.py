import torch 
import torch.nn as nn
import torch_geometric
import torch.optim as optimizer

device = torch.device('cuda:0')
input_size = 1
hidden_size, latent_size = 1, 1

class TimeDistributed(nn.Module):
  def __init__(self, module, batch_first=False):
    super(TimeDistributed, self).__init__()
    self.module = module
    self.batch_first = batch_first

  def forward(self, x):

    if len(x.size()) <= 2:
      return self.module(x)

    # Squash samples and timesteps into a single axis
    x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

    y = self.module(x_reshape)

    # We have to reshape Y
    if self.batch_first:
      y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
    else:
      y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

    return y


## 인코더
class Encoder2(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim):
    super(Encoder2, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim1  = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim1,
      num_layers=1,
      batch_first=True
    )
    

  def forward(self, x):
    #x = x.reshape((1, self.seq_len, self.n_features))
    x, (hidden_n, _) = self.rnn1(x)
    return hidden_n.reshape((self.n_features, self.embedding_dim))

## 디코더
class Decoder2(nn.Module):
  def __init__(self, seq_len, input_dim, n_features):
    super(Decoder2, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim1, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size = input_dim,
      hidden_size=self.input_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.output_layer = nn.Linear(self.hidden_dim1, n_features)
    self.timedist = TimeDistributed(self.output_layer)
  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    #x = x.reshape((self.n_features, self.seq_len, self.input_dim))
    x, (hidden_n, cell_n) = self.rnn1(x)
    #x = x.reshape((self.seq_len, self.hidden_dim1))
    return self.timedist(x)
class RecurrentAutoencoder2(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim = 1):
    super(RecurrentAutoencoder2, self).__init__()
    self.encoder = Encoder2(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder2(seq_len, embedding_dim, n_features).to(device)
  def forward(self, x):
    encoded = self.encoder(x)
    x = self.decoder(encoded)
    return encoded,x
        