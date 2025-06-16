import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
class Att(nn.Module):
    def __init__(self, hidden_size, attention_size, num_layers):
        super(Att, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.num_layers = num_layers

        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * num_layers, attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size))
        nn.init.constant_(self.w_omega, 0.001)
        nn.init.constant_(self.u_omega, 0.001)

    def forward(self, x, gru_output):
        device = x.device
        mask = torch.sign(torch.abs(torch.sum(x, axis=-1))).to(device)
        attn_tanh = torch.tanh(torch.matmul(gru_output, self.w_omega))
        attn_hidden_layer = torch.matmul(attn_tanh, self.u_omega)
        paddings = torch.ones_like(mask) * (-10e8)
        attn_hidden_layer = torch.where(torch.eq(mask, 0), paddings, attn_hidden_layer)
        alphas = F.softmax(attn_hidden_layer, 1)
        attn_output = torch.sum(gru_output * torch.unsqueeze(alphas, -1), 1)
        return attn_output
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads,batch_first=True)

    def forward(self, query_features, key_value_features):
        attn_output, _ = self.multihead_attn(query_features, key_value_features, key_value_features)
        return attn_output     
    
class Model(nn.Module):

    def __init__(self, cfg):
        super(Model,self).__init__()
        self.cfg = cfg
        text_feat_dim = cfg["model"]["text_feat_dim"]
        audio_feat_dim = cfg["model"]["audio_feat_dim"]
        hidden_dim = cfg["model"]["hidden_dim"]
        num_classes = cfg["model"]["num_classes"]
        num_heads = cfg["model"]["num_heads"]
        self.num_layers = cfg["model"]["num_layers"]
        self.direction = cfg["model"]["direction"]
        self.hidden_size1 = cfg["model"]["hidden_size1"]
        self.hidden_size2 = cfg["model"]["hidden_size2"]
        self.attention_size = cfg["model"]["attention_size"]
        self.lstm1 = nn.LSTM(text_feat_dim, self.hidden_size1, self.num_layers, batch_first=True, bidirectional=True)
        self.lstm11 = nn.LSTM(text_feat_dim, self.hidden_size1, self.num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(audio_feat_dim, self.hidden_size2, self.num_layers, batch_first=True, bidirectional=True)
        self.lstm22 = nn.LSTM(audio_feat_dim, self.hidden_size2, self.num_layers, batch_first=True, bidirectional=True)

        self.Att1 = Att(self.hidden_size1, self.attention_size, self.num_layers)
        self.Att11 = Att(self.hidden_size1, self.attention_size, self.num_layers)
        self.Atts1 = Att(self.hidden_size1, self.attention_size, self.num_layers)
        self.Att2 = Att(self.hidden_size2, self.attention_size, self.num_layers)
        self.Att22 = Att(self.hidden_size2, self.attention_size, self.num_layers)
        self.Attt2 = Att(self.hidden_size2, self.attention_size, self.num_layers)


        self.st = nn.Sequential(
            nn.Linear((self.hidden_size1 + self.hidden_size2) * 6, 128),
            nn.ReLU())
        self.ss = nn.Linear(self.direction * self.hidden_size1, num_classes)
        self.tt = nn.Linear(self.direction * self.hidden_size2, num_classes)
        self.c_fc = nn.Linear(128, num_classes)
        self.cross_attn1 = CrossAttention(self.hidden_size1*2, num_heads)
        self.cross_attn2 = CrossAttention(self.hidden_size2*2, num_heads)

    def forward(self,text_feat,audio_feat):
        device = audio_feat.device        
        audio_feat = audio_feat.to(device)
        text_feat = text_feat.to(device)
        h1 = torch.zeros(self.num_layers * self.direction, text_feat.size(0),self.hidden_size1).to(device)
        h2 = torch.zeros(self.num_layers * self.direction, audio_feat.size(0),self.hidden_size2).to(device)
        h11 = torch.zeros(self.num_layers * self.direction, text_feat.size(0),self.hidden_size1).to(device)
        h22 = torch.zeros(self.num_layers * self.direction, audio_feat.size(0),self.hidden_size2).to(device)
        c1 = torch.zeros(self.num_layers * self.direction, text_feat.size(0),self.hidden_size1).to(device)
        c2 = torch.zeros(self.num_layers * self.direction, audio_feat.size(0),self.hidden_size2).to(device)
        c11 = torch.zeros(self.num_layers * self.direction, text_feat.size(0),self.hidden_size1).to(device)
        c22 = torch.zeros(self.num_layers * self.direction, audio_feat.size(0),self.hidden_size2).to(device)

        out1, _ = self.lstm1(text_feat, (h1,c1))
        out2, _ = self.lstm2(audio_feat, (h2,c2))
        out11, _ = self.lstm11(text_feat, (h11,c11))
        out22, _ = self.lstm22(audio_feat, (h22,c22))

        outa1 = self.Att1(text_feat, out1)
        outa2 = self.Att2(audio_feat, out2)
        outa11 = self.Att11(text_feat, out11)
        outa22 = self.Att22(audio_feat, out22)       
        attn_output1 = self.cross_attn1(out1, out22)
        attn_output2 = self.cross_attn2(out2, out11)
        outatt1 = self.Atts1(text_feat, attn_output1)
        outatt2 = self.Attt2(audio_feat, attn_output2)

        con1 = torch.cat((outa1,outa2),1)
        con2 = torch.cat((outa11,outa22),1)
        con3 = torch.cat((outatt1,outatt2),1)
        c_all_c = torch.cat((con1,con2,con3),dim=1)
        out = self.st(c_all_c) 

        output = self.c_fc(out)
        return output,outa1, outa2, outa11, outa22,self.ss(outa1), self.tt(outa2)