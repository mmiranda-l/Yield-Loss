
import torch
import torch.nn as nn
import torch.utils.data
import os
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot product attention.
    Inspired from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """

    def __init__(self, hidden_size, len_max_seq, dropout: float=0.2):
        super().__init__()
        self.len_max_seq = len_max_seq
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.query = nn.Parameter(
            torch.zeros(1, 1, self.len_max_seq)
        )  # learnable query
        self.W_k = nn.Linear(hidden_size, len_max_seq)
        self.W_v = nn.Linear(hidden_size, len_max_seq)

    def forward(self, x):        
        
        keys = self.W_k(x)  # (b, time, dmodel)
        values = self.W_v(x)  # (b, time, dmodel)

        query = self.query.expand(keys.shape[0], -1, -1)

        scale_factor = 1 / math.sqrt(self.len_max_seq)
        attn_weight = torch.bmm(query, keys.transpose(1, 2)) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = self.dropout(attn_weight)
        output = torch.bmm(attn_weight, values)

        return output.view(output.shape[0], output.shape[-1])

class LSTM_Model_Seq(nn.Module):
    def __init__(self, in_channels=12, len_max_seq=24, hidden_size=128, num_layers=2, dropout=0.3):
        self.in_channels = in_channels
        self.len_max_seq = len_max_seq
        super(LSTM_Model_Seq, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )  # bidirectional not for forecasting
        self.linear = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            #nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Flatten(start_dim=1),
        )

    def _logits(self, x):
        # b,t,d
        out, (h_n, c_n) = self.lstm(x)
        # out: b, t, d_model
        # h_n: num_layers, b, d_model
        out = self.linear(out)

        return out

    def forward(self, x):
        # 32, 24, 12
        # 24, 12
        logits = self._logits(x)
        return logits

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop("model_state", snapshot)
        self.load_state_dict(model_state)
        return snapshot
    

class LSTM_Model_Loss_Seq(LSTM_Model_Seq):
    def __init__(self, in_channels=12, len_max_seq=24, hidden_size=128, num_layers=2, dropout=0.3, temp_attn=True, out=1):
        super().__init__(in_channels, len_max_seq, hidden_size, num_layers, dropout)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention = ScaledDotProductAttention(hidden_size, len_max_seq) if temp_attn else nn.Identity()
        if temp_attn == True: 
            hidden_size = len_max_seq
            out = len_max_seq

        self.linear = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            #nn.LeakyReLU(),
            nn.Linear(hidden_size, out),
            nn.Flatten(start_dim=1),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            #nn.LeakyReLU(),
            nn.Linear(hidden_size, out),
            nn.Flatten(start_dim=1),
        )

    def _logits(self, x):
        # b,t,d
        rnn_out, (h_n, c_n) = self.lstm(x)
        # out: b, t, d_model
        # h_n: num_layers, b, d_model
        #keys = self.key(rnn_out)
        #values = self.value(rnn_out)
        #attention_scores, _ = self.attention(keys.permute(0, -1, 1), values.permute(0,-1,1))
        attention_scores = self.attention(rnn_out)
        out_1 = self.linear(attention_scores)
        out_2 = self.linear2(attention_scores)
        return out_1, out_2
    
class LSTM_Model(nn.Module):
    def __init__(self, in_channels=12, len_max_seq=24, hidden_size=128, num_layers=2, dropout=0.3):
        self.in_channels = in_channels
        self.len_max_seq = len_max_seq
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=hidden_size * 2, out_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1),
            nn.Flatten(start_dim=1),
            nn.Linear(len_max_seq, 12),
            nn.BatchNorm1d(num_features=12),
            nn.LeakyReLU(),
            nn.Linear(12, 1),
            nn.ReLU(),
        )

    def _logits(self, x):
        # b,t,d
        out, (h_n, c_n) = self.lstm(x)
        # out: b, t, d_model*2
        # h_n: 2*num_layers, b, d_model
        out = self.linear(out)
        out = out.squeeze(-1)
        # out = self.final_linear(out).squeeze(-1)
        return out

    def forward(self, x):
        # 32, 24, 12
        # 24, 12
        logits = self._logits(x)
        return logits

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to " + path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state, **kwargs), path)

    def load(self, path):
        print("loading model from " + path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop("model_state", snapshot)
        self.load_state_dict(model_state)
        return snapshot

class LSTM_Model_Loss(LSTM_Model):
    def __init__(self, in_channels=12, len_max_seq=24, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__(in_channels, len_max_seq, hidden_size, num_layers, dropout)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attention = ScaledDotProductAttention(hidden_size, dropout)
        
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=len_max_seq, out_features=len_max_seq),
            #nn.LeakyReLU(),
            nn.Linear(len_max_seq, 1),
            nn.Flatten(start_dim=1),
        )
        
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=len_max_seq, out_features=len_max_seq),
            #nn.LeakyReLU(),
            nn.Linear(len_max_seq, 1),
            nn.Flatten(start_dim=1),
        )
    
    def _logits(self, x):
        # b,t,d
        rnn_out, (h_n, c_n) = self.lstm(x)
        keys = self.key(rnn_out)
        values = self.value(rnn_out)
        attention_scores = self.attention(keys.permute(0, -1, 1), values.permute(0,-1,1))
        # out: b, t, d_model*2
        # h_n: 2*num_layers, b, d_model
        out_1 = self.linear(attention_scores)
        out_2 = self.linear2(attention_scores)
        return out_1, out_2

    def forward(self, x):
        # 32, 24, 12
        # 24, 12
        logits = self._logits(x)
        return logits


if __name__ == "__main__":
    batch_size = 4
    time_steps = 24
    num_channels = 12
    sample = torch.ones(size=[batch_size, time_steps, num_channels])
    print(sample.shape)
    model = LSTM_Model_Loss_Seq(num_layers=2, len_max_seq=time_steps, temp_attn=True)
    print(model)
    eta, ky = model(sample)
    print(eta.shape)
    num_params = sum([m.numel() for m in model.parameters()])
    print(num_params)
