from data_process import *
from model import Model
import utils
import os
import json
import torch
import torch.utils.data as DATA
import numpy as np


# 参数
param = utils.param

# 生成 mention 的候选实体字典
if os.path.exists('../data/generated/cand.json'):
    with open('../data/generated/cand.json', 'r', encoding='utf8') as f:
        jsonstr = ''.join(f.readlines())
        cand_dic = json.loads(jsonstr)
    with open('../data/generated/entity.json', 'r', encoding='utf8') as f:
        jsonstr = ''.join(f.readlines())
        ent_dic = json.loads(jsonstr)
else:
    cand_dic, ent_dic = GenerateCand('kb.json')

# 生成训练、验证、测试的文本数据
if not os.path.exists('../data/generated/train_data.txt'):
    GeneratePairwaiseSample('train.json', cand_dic, ent_dic, is_train=True)
if not os.path.exists('../data/generated/dev_data.txt'):
    GeneratePairwaiseSample('dev.json', cand_dic, ent_dic, is_train=False)
if not os.path.exists('../data/generated/test_data.txt'):
    GeneratePairwaiseSample('test.json', cand_dic, ent_dic, is_train=False)

# matrix 向量数组；vocab 包含 vocab["w2i"]: word2idx、vocab["i2w"]：idx2word；向量维度，字词数
if not os.path.exists('../data/pretrain_data/matrix.npy'):
    matrix, vocab, vec_dim, vocab_size = utils.loadWord2Vec("../data/pretrain_data/word2vec.iter5")
else:
    matrix = np.load('../data/pretrain_data/matrix.npy')
    with open('../data/pretrain_data/vocab.json', 'r', encoding='utf8') as f:
        jsonstr = ''.join(f.readlines())
        vocab = json.loads(jsonstr)

# 类型2标签字典
type2label = utils.type2label

# 数据编码
data_encoder = DataEncoder(vocab["w2i"], type2label)
if not os.path.exists('../data/generated/train.csv'):
    data_encoder.data_encode("../data/generated/train_data.txt", is_train=True)
if not os.path.exists('../data/generated/dev.csv'):
    data_encoder.data_encode("../data/generated/dev_data.txt", is_train=False)
if not os.path.exists('../data/generated/test.csv'):
    data_encoder.data_encode("../data/generated/test_data.txt", is_train=False)

# 构建数据集加载接口
train_set = DataSet("../data/generated/train_part.csv", is_train=True)
dev_set = DataSet("../data/generated/dev.csv", is_train=False)
test_set = DataSet("../data/generated/test.csv", is_train=False)

# dataloader
train_loader = DATA.DataLoader(train_set, batch_size=param["batch"], collate_fn=utils.collate_fn_train, drop_last=True)
dev_loader = DATA.DataLoader(dev_set, batch_size=param["batch"], collate_fn=utils.collate_fn_test, drop_last=True)
test_loader = DATA.DataLoader(test_set, batch_size=param["batch"], collate_fn=utils.collate_fn_test, drop_last=True)

# 实例化模型
model = Model(matrix, param) # martix 是预训练的 word embedding
model.to(param["device"])

# 优化器
model.embedding.weight.requires_grad = False    # 冻结 embedding 层的参数
optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=param["lr"])

# 联合损失函数
rank_loss_fn = torch.nn.BCELoss()
classify_loss_fn = torch.nn.CrossEntropyLoss()

param["epoch"] = 10
param["loss_w"] = 0
device = param["device"]
total_step = len(train_loader)
train_accuracy = []
dev_accuracy = []

# 训练模型
for epoch in range(param["epoch"]):
    train_res = {}
    # train
    for i, data in enumerate(train_loader):
        id_list, query, offset, cand1_desc, cand2_desc, label, ent_type, seq_len = data
        # move the data to the device
        query = query.to(device)
        cand1_desc = cand1_desc.to(device)
        cand2_desc = cand2_desc.to(device)
        label = label.to(device)
        ent_type = ent_type.to(device)
        # forward
        pre_label, pre_type = model.forward(query, offset, cand1_desc, cand2_desc, seq_len)
        # loss
        rank_loss = rank_loss_fn(pre_label, label)
        type_loss = classify_loss_fn(pre_type, ent_type)
        loss = rank_loss * param["loss_w"] + type_loss * (1 - param["loss_w"])
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_res = utils.record(train_res, id_list, torch.softmax(pre_label, dim=-1), pre_type, label)
        if i % 1000 == 0:
            print('Epoch [{}/{}], Step [{}/{}]'.format(epoch, param["epoch"], i, total_step))
            print("Loss: ", loss.item(), "rank_loss: ", rank_loss.item(), "type_loss: ", type_loss.item())
    
    accuracy = utils.Accuracy(train_res)
    train_accuracy.append(accuracy)
    print("train accuracy: ", accuracy)
    dev_res = {}
    # evalue
    with torch.no_grad():
        for  i, data in enumerate(dev_loader):
            id_list, query, offset, cand_desc, seq_len = data
            # move the data to the device
            query = query.to(device)
            cand_desc = cand_desc.to(device)
            # forward
            pre_label, pre_type = model.predict(query, offset, cand_desc, seq_len)
            # loss
            rank_loss = rank_loss_fn(pre_label, label)
            type_loss = classify_loss_fn(pre_type, ent_type)
            loss = rank_loss * param["loss_w"] + type_loss * (1 - param["loss_w"])
            # 记录预测结果
            dev_res = utils.record(dev_res, id_list, torch.softmax(pre_label, dim=-1), pre_type)
        accuracy = utils.Accuracy(dev_res)
        dev_accuracy.append(accuracy)
        print("dev accuracy: ", accuracy)
        torch.cuda.empty_cache()
    
    # 保存断点
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        "train_accuracy": train_accuracy,
        "test_accuracy": dev_accuracy
    }
    if not os.path.exists("./models/checkpoint"):
        os.mkdir("./models/checkpoint")
    torch.save(checkpoint, './models/checkpoint/ckpt_best_%s.pth' % (str(epoch)))
               
