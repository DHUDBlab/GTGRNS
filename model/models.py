"""
-*- coding: utf-8 -*-
@Author : JeikoLiu
@Institution : DHU/DBLab
@UpdateTime : 2024/12/06 21:52
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv

class GTGRNS(nn.Module):

    def __init__(self, inputDim, positionSize, num_layers, num_head, aggregator, normalize):
        super(GTGRNS, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.positionSize = positionSize

        # batch_first 表示输入数据的形状 -- [batch_size, seq_len, features]
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=positionSize, nhead=num_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.position_embedding = nn.Embedding(2, positionSize)
        self.sage_encoder = SAGEConv(inputDim[1], positionSize, aggregator, normalize)
        '''
        self.attention_layer1 = GATConv(expression_data_shape[1], hidden_dim1, heads=att_head1, concat=True, negative_slope=0.2)
        self.attention_layer2 = GATConv(hidden_dim1*att_head1, hidden_dim2, heads=att_head2, concat=True, negative_slope=0.2)
        '''

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1536, 1024) # original = 1536
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bc3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)
        self.outF = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def Sage_forward(self, gene_expr, edge_index):
        expr_embedding = self.sage_encoder(gene_expr, edge_index).to(self.device)
        expr_embedding = F.relu(expr_embedding)
        return expr_embedding

    def GAT_forward(self, gene_expr, edge_index):
        expr_embedding = self.attention_layer1(gene_expr, edge_index).to(self.device)
        expr_embedding = F.elu(expr_embedding)
        # expr_embedding = self.attention_layer2(expr_embedding, edge_index).to(self.device)
        # expr_embedding = F.elu(expr_embedding)
        return expr_embedding

    def forward(self, gene_pair_index, expr_embedding):
        bs = gene_pair_index.shape[0]  # batch_size（每次处理的样本数量）

        # 首先创建包含重复bs次的交替的 0 和 1 的一维 Tensor，然后进行reshape
        position = torch.Tensor([0, 1] * bs).reshape(bs, -1).to(torch.int32)
        position = position.to(self.device)
        p_e = self.position_embedding(position)
        pair_list = []
        for i in range(gene_pair_index.shape[0]):
            tf_embedding = expr_embedding[gene_pair_index[i][0]].view(1, self.embed_size).to(self.device)
            target_embedding = expr_embedding[gene_pair_index[i][1]].view(1, self.embed_size).to(self.device)
            tmp = torch.cat((tf_embedding, target_embedding), dim=0)
            pair_list.append(tmp)
        out_expr_e = torch.stack(pair_list)

        input = torch.add(out_expr_e, p_e)
        out = self.transformer_encoder(input)
        out = self.flatten(out)

        out = self.fc1(out)
        out = self.dropout(out)
        out = self.outF(out)
        p = out.unsqueeze(1)
        p = self.pool(p)
        p = p.squeeze(1)

        out = self.fc2(out)
        out = self.dropout(out)
        out = self.outF(out)
        out = self.fc3(out) + p
        # out = out + torch.randn_like(out) * 0.1
        out = self.dropout(out)
        out = self.outF(out)
        out = self.fc4(out)

        score = self.sigmoid(out)
        return score