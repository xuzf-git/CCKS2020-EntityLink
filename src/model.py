import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


class Model(nn.Module):
    def __init__(self, embd, param):
        super(Model, self).__init__()
        self.param = param
        # 加载预训练的词向量
        embd = torch.from_numpy(embd)
        self.embedding = nn.Embedding.from_pretrained(embd)
        # # 将embedding层设置为 “不计算梯度，不进行更新”
        # for p in self.parameters():
        #     p.requires_grad=False
        # 提取查询文本序列（query）特征信息的BiLSTM层
        self.bilstm_query = nn.LSTM(input_size=param["emb_dim"] + 1,
                                    hidden_size=param["hidden_dim"],
                                    batch_first=True,
                                    bidirectional=True,)
        # 提取候选实体描述序列（cand_desc）特征信息的BiLSTM层
        self.bilstm_cand = nn.LSTM(input_size=param["emb_dim"],
                                   hidden_size=param["hidden_dim"],
                                   batch_first=True,
                                   bidirectional=True,)
        # 用于候选实体排序的全连接层
        self.fc_rank = nn.Sequential(
            nn.Linear(param["hidden_dim"] * 2, param["hidden_dim_fc"]),
            nn.ReLU(inplace=True),
            nn.Linear(param["hidden_dim_fc"], 1),
            nn.Sigmoid()
        )
        # 用于实体分类的全连接层
        self.fc_classify = nn.Sequential(
            nn.Linear(param["hidden_dim"] * 2, param["type_num"]),
        )

    def init_hidden(self):
        """ 生成BiLSTM的初始化状态 """
        hidden_state = torch.zeros(1 * 2, self.batch, self.param["hidden_dim"]).to(self.param["device"])
        cell_state = torch.zeros(1 * 2, self.batch, self.param["hidden_dim"]).to(self.param["device"])
        return hidden_state, cell_state

    def Attention(self, q, k, v, scale=None, attn_mask=None):
        """
        desc: scaled dot-product attention
        q: [batch, timestep_q, dim_q]
        k: [batch, timestep_k, dim_k]
        v: [batch, timestep_v, dim_v]
        scale: 缩放因子
        attn_mask: Masking 张量 [batch, timestep_q, timestep_k]
        context: [batch, dim_v]
        """
        attention = torch.bmm(q, k.transpose(1,2))
        if scale is None:
            scale = (1 / torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32))).item()
        attention = attention * scale
        if attn_mask:
            # 将需要 mask 的地方设为负无穷
            attention = attention.masked_fill(attn_mask, -np.inf)
        attention = torch.softmax(attention, dim=2)
        context = torch.sum(torch.bmm(attention, v), dim=1)
        return context

    def bilstm_with_mask(self, seq, seq_len, is_query=False):
        """
        desc: BiLSTM with Mask
        """
        # pack padded
        seq = pack_padded_sequence(seq, seq_len, batch_first=True, enforce_sorted=False)
        unsorted_indices = seq.unsorted_indices
        init_h = self.init_hidden()
        # bilstm
        if is_query:
            seq_out, _ = self.bilstm_query(seq, init_h)
        else:
            seq_out, _ = self.bilstm_cand(seq, init_h)
        # pad pack
        seq_out = pad_packed_sequence(seq_out, batch_first=True, padding_value=1)
        seq_out = seq_out[0][unsorted_indices]
        return seq_out

    def forward(self, query, offset, cand1, cand2, seq_len):
        # embedding
        query = self.embedding(query)
        # 拼接 mention offset：
        # 对于每个输入样本（每个query中的每个实体），将query中实体的offset位置特征编码为：长度等于句子长度，且实体部分为1，
        # 非实体部分为0的特征向量，并且拼接到每个词向量的最后一维得到 batch * seq_len * 301 维度的向量序列
        self.batch = query.shape[0]
        pos = torch.zeros([query.shape[0], query.shape[1], 1])
        for i in range(query.shape[0]):
            pos[i][offset[i]][0] = 1.0
        pos = pos.to(self.param["device"])
        query = torch.cat((query, pos), dim=2)
        # 对候选实体描述进行编码
        cand1 = self.embedding(cand1)
        cand2 = self.embedding(cand2)
        # bilstm with mask
        query_out = self.bilstm_with_mask(query, seq_len[0], is_query=True)
        cand1_out = self.bilstm_with_mask(cand1, seq_len[1])
        cand2_out = self.bilstm_with_mask(cand2, seq_len[2])
        # attention
        score_type = self.Attention(query_out, query_out, query_out)
        score_cand11 = self.Attention(query_out, cand1_out, cand1_out)
        score_cand12 = self.Attention(cand1_out, query_out, query_out)
        score_cand21 = self.Attention(query_out, cand2_out, cand2_out)
        score_cand22 = self.Attention(cand2_out, query_out, query_out)
        # 使用pairwise模型分别计算两个候选实体的得分
        score_cand1 = self.fc_rank(score_cand11 + score_cand12)
        score_cand2 = self.fc_rank(score_cand21 + score_cand22)
        # 比较两候选实体得分的
        pred_rank = torch.sigmoid(score_cand1 - score_cand2)
        # 对实体提及的类型进行预测
        pred_type = self.fc_classify(score_type)
        return pred_rank.squeeze(), pred_type

    def predict(self, query, offset, cand, seq_len):
        # embedding
        query = self.embedding(query)
        # 拼接 mention offset：方法同 self.forward
        self.batch = query.shape[0]
        pos = torch.zeros([query.shape[0], query.shape[1], 1])
        for i in range(query.shape[0]):
            pos[i][offset[i]][0] = 1.0
        pos = pos.to(self.param["device"])
        query = torch.cat((query, pos), dim=2)
        # 对候选实体描述进行编码
        cand = self.embedding(cand)
        # bilstm with mask
        query_out = self.bilstm_with_mask(query, seq_len[0], is_query=True)
        cand_out = self.bilstm_with_mask(cand, seq_len[1])
        # attention
        score_type = self.Attention(query_out, query_out, query_out)
        score_cand1 = self.Attention(query_out, cand_out, cand_out)
        score_cand2 = self.Attention(cand_out, query_out, query_out)
        # 计算该候选实体得分
        pred_rank = self.fc_rank(score_cand1 + score_cand2)
        # 预测该实体类型
        pred_type = self.fc_classify(score_type)
        return pred_rank.squeeze(), pred_type

