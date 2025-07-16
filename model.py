import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# 带多头注意力的分层BiLSTM
class HBiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # 第一层BiLSTM
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 第二层BiLSTM
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        """返回单个输出张量，与原始接口兼容"""
        # 第一层BiLSTM
        x1, _ = self.lstm1(x)

        # 多头注意力
        attn_out, _ = self.self_attn(x1, x1, x1)
        x1 = self.attn_norm(x1 + attn_out)

        # 第二层BiLSTM
        x2, _ = self.lstm2(x1)
        return x2  # 只返回输出张量

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, leaky_slope=0.1):
        super(ResidualBlock, self).__init__()
        # 主路径：三个卷积层
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=leaky_slope)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=leaky_slope)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # 残差路径：1x1卷积调整维度
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        # 主路径
        out = self.leaky_relu1(self.bn1(self.conv1(x)))  # Conv1 → BN → LeakyReLU
        out = self.leaky_relu2(self.bn2(self.conv2(out)))  # Conv2 → BN → LeakyReLU
        out = self.bn3(self.conv3(out))  # Conv3 → BN（不加激活）

        # 处理长度不匹配（优先确保kernel_size为奇数）
        if out.size(2) != shortcut.size(2):
            min_len = min(out.size(2), shortcut.size(2))
            out = out[:, :, :min_len]
            shortcut = shortcut[:, :, :min_len]

        return F.leaky_relu(out + shortcut, negative_slope=0.1)  # 残差相加后统一激活

# SelfAttention 层
class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key=None, value=None):
        if key is None or value is None:
            key = value = query
        residual = query
        attn_output, _ = self.attention(query, key, value)
        attn_output = self.dropout(attn_output)
        return self.layernorm(residual + attn_output)

class MultiHopAttention(nn.Module):
    def __init__(self, d_model, nhead, num_hops):
        super(MultiHopAttention, self).__init__()
        self.attention_layers = nn.ModuleList([
            SelfAttention(d_model, nhead) for _ in range(num_hops)
        ])

    def forward(self, query, key_value):
        for att in self.attention_layers:
            query = att(query, key_value, key_value)
        return query

class MHDA_MHPA(nn.Module):
    def __init__(self, d_model, nhead, num_hops):
        super(MHDA_MHPA, self).__init__()
        self.multi_hop_attention = MultiHopAttention(d_model, nhead, num_hops)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, query, key_value):
        attn_out = self.multi_hop_attention(query, key_value)
        attn_out = self.dropout(self.fc(attn_out))
        return self.layernorm(attn_out + query)

class Hierarchical_Heterogeneous_att(nn.Module):
    def __init__(self, d_model, nhead, num_sa_layers, num_dta_layers, num_hops):
        super(Hierarchical_Heterogeneous_att, self).__init__()
        self.sa_drug = nn.ModuleList([
            SelfAttention(d_model, nhead) for _ in range(num_sa_layers)
        ])
        self.sa_target = nn.ModuleList([
            SelfAttention(d_model, nhead) for _ in range(num_sa_layers)
        ])
        self.mhta_layers = nn.ModuleList([
            MHDA_MHPA(d_model, nhead, num_hops) for _ in range(num_dta_layers)
        ])
        self.mhda_layers = nn.ModuleList([
            MHDA_MHPA(d_model, nhead, num_hops) for _ in range(num_dta_layers)
        ])

    def forward(self, drug_feat, target_feat):
        # (B, L, D)
        for sa in self.sa_drug:
            drug_feat = sa(drug_feat)
        for sa in self.sa_target:
            target_feat = sa(target_feat)

        for mhta, mhda in zip(self.mhta_layers, self.mhda_layers):
            target_feat = mhta(target_feat, drug_feat)
            drug_feat = mhda(drug_feat, target_feat)

        return drug_feat, target_feat

class HFEDTI(nn.Module):
    def __init__(self, hp, protein_MAX_LENGTH=1000, drug_MAX_LENGTH=100):
        super(HFEDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGTH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGTH
        self.protein_kernel = hp.protein_kernel

        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4

        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - sum(self.drug_kernel) + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - sum(self.protein_kernel) + 3

        self.mix_attention_head = 5

        # Embedding
        self.drug_embed = nn.Embedding(self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            ResidualBlock(self.dim, self.conv, self.drug_kernel[0], leaky_slope=0.1),
            ResidualBlock(self.conv, self.conv * 2, self.drug_kernel[1], leaky_slope=0.1),
            ResidualBlock(self.conv * 2, self.conv * 4, self.drug_kernel[2], leaky_slope=0.1)
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)

        self.Protein_CNNs = nn.Sequential(
            ResidualBlock(self.dim, self.conv, self.protein_kernel[0], leaky_slope=0.1),
            ResidualBlock(self.conv, self.conv * 2, self.protein_kernel[1], leaky_slope=0.1),
            ResidualBlock(self.conv * 2, self.conv * 4, self.protein_kernel[2], leaky_slope=0.1)
        )
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        self.drug_bilstm = HBiLSTM_Attention(
            input_dim=hp.conv * 4,  # 输入维度与原始一致
            hidden_dim=hp.attention_dim,  # 隐藏层维度与原始一致
            num_heads=4  # 可调节的超参数
        )
        self.protein_bilstm = HBiLSTM_Attention(
            input_dim=hp.conv * 4,
            hidden_dim=hp.attention_dim,
            num_heads=4
        )

        # 注意力层
        self.Hierarchical_Heterogeneous_att = Hierarchical_Heterogeneous_att(
            self.attention_dim, self.mix_attention_head, num_sa_layers=2, num_dta_layers=2, num_hops=3
        )

        # 全连接层
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc1 = nn.Linear(self.conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein):
        drugembed = self.drug_embed(drug).permute(0, 2, 1)  # [B, C, L]
        proteinembed = self.protein_embed(protein).permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed).permute(0, 2, 1)  # [B, L, D]
        proteinConv = self.Protein_CNNs(proteinembed).permute(0, 2, 1)

        drug_lstm = self.drug_bilstm(drugConv)  # [B, L, D]
        protein_lstm = self.protein_bilstm(proteinConv)  # [B, L, D]

        drug_att, protein_att = self.Hierarchical_Heterogeneous_att(drug_lstm, protein_lstm)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        drugConv = drugConv.permute(0, 2, 1)  # [B, D, L]
        proteinConv = proteinConv.permute(0, 2, 1)

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        # 全连接层
        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict

    def extract_features(self, drug, protein):
        """特征提取（与forward前半部分一致）"""
        drugembed = self.drug_embed(drug).permute(0, 2, 1)
        proteinembed = self.protein_embed(protein).permute(0, 2, 1)
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)
        drug_lstm= self.drug_bilstm(drugConv.permute(0, 2, 1))
        protein_lstm = self.protein_bilstm(proteinConv.permute(0, 2, 1))
        drug_QKV = drug_lstm.permute(1, 0, 2)
        protein_QKV = protein_lstm.permute(1, 0, 2)
        drug_att = self.Hierarchical_Heterogeneous_att(drug_QKV, protein_QKV, protein_QKV).permute(1, 2, 0)
        protein_att = self.Hierarchical_Heterogeneous_att(protein_QKV, drug_QKV, drug_QKV).permute(1, 2, 0)
        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)
        return torch.cat([drugConv, proteinConv], dim=1)


# casestudy
'''class HFEDTI(nn.Module):
    def __init__(self, hp, protein_MAX_LENGTH=1000, drug_MAX_LENGTH=100):
        super(HFEDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = drug_MAX_LENGTH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = protein_MAX_LENGTH
        self.protein_kernel = hp.protein_kernel

        self.drug_vocab_size = 65
        self.protein_vocab_size = 26
        self.attention_dim = hp.conv * 4

        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - sum(self.drug_kernel) + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - sum(self.protein_kernel) + 3

        self.mix_attention_head = 5

        # Embedding
        self.drug_embed = nn.Embedding(self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(self.protein_vocab_size, self.dim, padding_idx=0)

        # 残差卷积提取器
        self.Drug_CNNs = nn.Sequential(
            ResidualBlock(self.dim, self.conv, self.drug_kernel[0], leaky_slope=0.1),
            ResidualBlock(self.conv, self.conv * 2, self.drug_kernel[1], leaky_slope=0.1),
            ResidualBlock(self.conv * 2, self.conv * 4, self.drug_kernel[2], leaky_slope=0.1)
        )
        self.Protein_CNNs = nn.Sequential(
            ResidualBlock(self.dim, self.conv, self.protein_kernel[0], leaky_slope=0.1),
            ResidualBlock(self.conv, self.conv * 2, self.protein_kernel[1], leaky_slope=0.1),
            ResidualBlock(self.conv * 2, self.conv * 4, self.protein_kernel[2], leaky_slope=0.1)
        )

        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

        # BiLSTM + 注意力
        self.drug_bilstm = HBiLSTM_Attention(
            input_dim=hp.conv * 4,
            hidden_dim=hp.attention_dim,
            num_heads=4
        )
        self.protein_bilstm = HBiLSTM_Attention(
            input_dim=hp.conv * 4,
            hidden_dim=hp.attention_dim,
            num_heads=4
        )

        # 层次化跨模态交叉注意力
        self.Hierarchical_Heterogeneous_att = Hierarchical_Heterogeneous_att(
            self.attention_dim, self.mix_attention_head, num_sa_layers=2, num_dta_layers=2, num_hops=3
        )

        # === 新增：对齐维度 ===
        self.align_drug_att = nn.Conv1d(hp.attention_dim, hp.conv * 4, kernel_size=1)
        self.align_protein_att = nn.Conv1d(hp.attention_dim, hp.conv * 4, kernel_size=1)

        # 全连接层
        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.fc1 = nn.Linear(self.conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

    def forward(self, drug, protein):
        # Embedding
        drugembed = self.drug_embed(drug).permute(0, 2, 1)
        proteinembed = self.protein_embed(protein).permute(0, 2, 1)

        # 卷积提取
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        # BiLSTM + Attention
        drug_lstm = self.drug_bilstm(drugConv.permute(0, 2, 1))
        protein_lstm = self.protein_bilstm(proteinConv.permute(0, 2, 1))

        # 跨模态注意力
        drug_QKV = drug_lstm.permute(1, 0, 2)
        protein_QKV = protein_lstm.permute(1, 0, 2)
        drug_att = self.Hierarchical_Heterogeneous_att(drug_QKV, protein_QKV, protein_QKV).permute(1, 2, 0)
        protein_att = self.Hierarchical_Heterogeneous_att(protein_QKV, drug_QKV, drug_QKV).permute(1, 2, 0)

        # === 对齐维度 ===
        drug_att = self.align_drug_att(drug_att)       # -> [batch, conv*4, seq_len]
        protein_att = self.align_protein_att(protein_att)

        # 残差融合
        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5

        # 池化
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        # 分类
        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict

    def extract_features(self, drug, protein):
        drugembed = self.drug_embed(drug).permute(0, 2, 1)
        proteinembed = self.protein_embed(protein).permute(0, 2, 1)
        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)
        drug_lstm = self.drug_bilstm(drugConv.permute(0, 2, 1))
        protein_lstm = self.protein_bilstm(proteinConv.permute(0, 2, 1))
        drug_QKV = drug_lstm.permute(1, 0, 2)
        protein_QKV = protein_lstm.permute(1, 0, 2)
        drug_att = self.Hierarchical_Heterogeneous_att(drug_QKV, protein_QKV, protein_QKV).permute(1, 2, 0)
        protein_att = self.Hierarchical_Heterogeneous_att(protein_QKV, drug_QKV, drug_QKV).permute(1, 2, 0)

        # === 对齐维度 ===
        drug_att = self.align_drug_att(drug_att)
        protein_att = self.align_protein_att(protein_att)

        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)
        return torch.cat([drugConv, proteinConv], dim=1)
'''
