from tqdm import tqdm
import numpy as np
import torch

# 参数字典
param = {}
param["batch"] = 64
param["epoch"] = 10
param["lr"] = 0.02
param["loss_w"] = 0.6
param["type_num"] = 24
param["emb_dim"] = 300
param["hidden_dim"] = 150
param["hidden_dim_fc"] = 150
param["resume"] = True
param["use_cuda"] = False # torch.cuda.is_available()
param["device"] = 'cpu' # torch.device('cuda' if param["use_cuda"] else 'cpu')

# 构建类型到标签的映射字典
type2label = {
    "Other": 0,
    "Person": 1,
    "Work": 2,
    "Culture": 3,
    "Organization": 4,
    "VirtualThings": 5,
    "Location": 6,
    "Education": 7,
    "Website": 8,
    "Software": 9,
    "Game": 10,
    "Medicine": 11,
    "Natural&Geography": 12,
    "Biological": 13,
    "Event": 14,
    "Food": 15,
    "Disease&Symptom": 16,
    "Constellation": 17,
    "Time&Calendar": 18,
    "Brand": 19,
    "Vehicle": 20,
    "Awards": 21,
    "Law&Regulation": 22,
    "Diagnosis&Treatment": 23
}

lable2type = [
    "Other",
    "Person",
    "Work",
    "Culture",
    "Organization",
    "VirtualThings",
    "Location",
    "Education",
    "Website",
    "Software",
    "Game",
    "Medicine",
    "Natural&Geography",
    "Biological",
    "Event",
    "Food",
    "Disease&Symptom",
    "Constellation",
    "Time&Calendar",
    "Brand",
    "Vehicle",
    "Awards",
    "Law&Regulation",
    "Diagnosis&Treatment",
]

def loadWord2Vec(path):
    """加载词向量"""
    vocab_size, size = 0, 0
    vocab = {}
    vocab["i2w"], vocab["w2i"] = [], {}
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        first_line = True
        for line in tqdm(f, desc='Build vocab'):
            if first_line:
                first_line = False
                vocab_size = int(line.strip().split()[0]) + 2
                size = int(line.rstrip().split()[1])
                matrix = np.zeros(shape=(vocab_size, size), dtype=np.float32)
                vocab["w2i"]["<unk>"] = count
                vocab["w2i"]["<pad>"] = count + 1
                matrix[1, :] = np.array([1.0] * size)
                count += 2
                continue
            vec = line.strip().split()
            if not vocab["w2i"].__contains__(vec[0]):
                vocab["w2i"][vec[0]] = count
                matrix[count, :] = np.array([float(x) for x in vec[1:]])
                count += 1
    for w, i in vocab["w2i"].items():
        vocab["i2w"].append(w)
    # matrix 向量数组；vocab 包含 vocab["w2i"]: word2idx、vocab["i2w"]：idx2word；向量维度，字词数
    # matrix, vocab, vec_dim, vocab_size = loadWord2Vec("../data/pretrain_data/word2vec.iter5")
    return matrix, vocab, size, len(vocab["i2w"])

def collate_fn_train(batch):
    """dataloader 预处理函数参数"""
    max_len_query = 0
    max_len_cand1 = 0
    max_len_cand2 = 0
    batch_size = len(batch)
    len_seq_query,len_seq_cand1, len_seq_cand2 = [], [], []
    for each in batch:
        len_seq_query.append(len(each[1]))
        len_seq_cand1.append(len(each[3]))
        len_seq_cand2.append(len(each[4]))
        if len(each[1]) > max_len_query:
            max_len_query = len(each[1])
        if len(each[3]) > max_len_cand1:
            max_len_cand1 = len(each[3])
        if len(each[4]) > max_len_cand2:
            max_len_cand2 = len(each[4])
    padd_query = torch.LongTensor()
    padd_cand1 = torch.LongTensor()
    padd_cand2 = torch.LongTensor()
    id_list, offset, label, ent_type = [], [], [], []
    # 静态 padding 每个 text 序列到 batch 内最长
    for each in batch:
        tmp1 = torch.ones(max_len_query - len(each[1]), dtype=torch.long)
        tmp2 = torch.ones(max_len_cand1 - len(each[3]), dtype=torch.long)
        tmp3 = torch.ones(max_len_cand2 - len(each[4]), dtype=torch.long)
        padd_query = torch.cat([padd_query, torch.cat([each[1], tmp1])], dim=0)
        padd_cand1 = torch.cat([padd_cand1, torch.cat([each[3], tmp2])], dim=0)
        padd_cand2 = torch.cat([padd_cand2, torch.cat([each[4], tmp3])], dim=0)
        id_list.append(each[0])
        offset.append(each[2])
        label.append(each[5])
        ent_type.append(each[6])
    padd_query = padd_query.view(batch_size, -1)
    padd_cand1 = padd_cand1.view(batch_size, -1)
    padd_cand2 = padd_cand2.view(batch_size, -1)
    label = torch.tensor(label, dtype=torch.float)
    ent_type = torch.tensor(ent_type, dtype=torch.long)
    # 变长序列 query, cand1_desc, cand2desc 的序列长度
    seq_len = (len_seq_query, len_seq_cand1, len_seq_cand2)
    return id_list, padd_query, offset, padd_cand1, padd_cand2, label, ent_type, seq_len

def collate_fn_test(batch):
    """dataloader 预处理函数参数"""
    max_len_query = 0
    max_len_cand = 0
    batch_size = len(batch)
    len_seq_query,len_seq_cand = [], []
    for each in batch:
        len_seq_query.append(len(each[1]))
        len_seq_cand.append(len(each[3]))
        if len(each[1]) > max_len_query:
            max_len_query = len(each[1])
        if len(each[3]) > max_len_cand:
            max_len_cand = len(each[3])
    padd_query = torch.LongTensor()
    padd_cand = torch.LongTensor()
    id_list, offset = [], []
    # 静态 padding 每个 text 序列到 batch 内最长
    for each in batch:
        tmp1 = torch.ones(max_len_query - len(each[1]), dtype=torch.long)
        tmp2 = torch.ones(max_len_cand - len(each[3]), dtype=torch.long)
        padd_query = torch.cat([padd_query, torch.cat([each[1], tmp1])], dim=0)
        padd_cand = torch.cat([padd_cand, torch.cat([each[3], tmp2])], dim=0)
        id_list.append(each[0])
        offset.append(each[2])
    padd_query = padd_query.view(batch_size, -1)
    padd_cand = padd_cand.view(batch_size, -1)
    # 变长序列 query, cand_desc 的序列长度
    seq_len = (len_seq_query, len_seq_cand)
    return id_list, padd_query, offset, padd_cand, seq_len

def record(result, id_list, pre_label, pre_type, label=None):
    """记录预测结果"""
    if pre_label.dim() == 0:
        pre_label = pre_label.view([1])
    # train
    if len(id_list[0]) == 6: 
        for i in range(len(id_list)):
            text_id, mention_id, cand1_id, cand2_id, golden_type, golden_id = id_list[i]
            if text_id not in result:
                result[text_id] = {}
            if mention_id not in result[text_id]:
                result[text_id][mention_id] ={
                    'golden_id':golden_id, 'golden_type':golden_type, 'pre_type':pre_type[i], 
                }
                if pre_label[i] - label[i] < 0.5 and pre_label[i] - label[i] > -0.5:
                    result[text_id][mention_id]['pre_id'] = golden_id
                else:
                    result[text_id][mention_id]['pre_id'] = 'NIL'
            else:
                result[text_id][mention_id]['pre_type'] += torch.clone(pre_type[i])
                if pre_label[i] - label[i] < 0.5 and pre_label[i] - label[i] > -0.5:
                    continue
                else:
                    result[text_id][mention_id]['pre_id'] = 'NIL'
    # eval or test
    elif len(id_list[0]) == 5:
        for i in range(len(id_list)):
            text_id, mention_id, cand_id, golden_type, golden_id = id_list[i]
            if text_id not in result:
                result[text_id] = {}
            if mention_id not in result[text_id]:
                result[text_id][mention_id] = {
                    'golden_id':golden_id, 'golden_type':golden_type, 'pre_id': cand_id, 
                    'pre_type':pre_type[i], 'pre_id_score': pre_label[i]
                }
            else:
                result[text_id][mention_id]['pre_type'] += torch.clone(pre_type[i])
                if pre_label[i] > result[text_id][mention_id]['pre_id_score']:
                    result[text_id][mention_id]['pre_id_score'] = pre_label[i]
                    result[text_id][mention_id]['pre_id'] = cand_id
    return result

def Accuracy(result):
    """计算预测结果的 Accuracy: (预测正确数 / 预测总数)"""
    right = 0
    total = 0
    for i in result.items():
        text_id = i[0]
        mentions = i[1]
        total += len(mentions)
        for j in mentions.items():
            mention_id = j[0]
            golden_id = j[1]['golden_id']
            golden_type = j[1]['golden_type']
            pre_id = j[1]['pre_id']
            pre_type = j[1]['pre_type'].argmax().item()
            if pre_id.isdigit():
                if pre_id == golden_id:
                    right += 1
            else:
                if golden_type == str(pre_type) and golden_id == 'NIL':
                    right += 1
    accuracy= right / total
    return accuracy