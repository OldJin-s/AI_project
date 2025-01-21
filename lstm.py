import torch
import torch.nn as nn

class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMmodel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM 레이어
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size ,num_layers=2, batch_first=True)
        
        # 출력 레이어
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, 3),
            nn.Tanh()
        )
    
    def forward(self, x):
        # LSTM 레이어 통과
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out은 모든 시점의 출력
        
        # 마지막 시점의 출력만 사용
        last_hidden_state = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # 출력 레이어 통과
        output = self.linear(last_hidden_state)
        return output
