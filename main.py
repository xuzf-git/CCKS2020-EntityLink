from src.data_process import *
from src.model import Model
import utils
import os
import re
import json
import numpy as np

# matrix 向量数组；vocab 包含 vocab["w2i"]: word2idx、vocab["i2w"]：idx2word；向量维度，字词数
matrix = np.load('./data/pretrain_data/matrix.npy')
with open('./data/pretrain_data/vocab.json', 'r', encoding='utf8') as f:
    jsonstr = ''.join(f.readlines())
    vocab = json.loads(jsonstr)

# 生成 mention 的候选实体字典
if os.path.exists('./data/generated/cand.json'):
    with open('./data/generated/cand.json', 'r', encoding='utf8') as f:
        jsonstr = ''.join(f.readlines())
        cand_dic = json.loads(jsonstr)
    with open('./data/generated/entity.json', 'r', encoding='utf8') as f:
        jsonstr = ''.join(f.readlines())
        ent_dic = json.loads(jsonstr)
else:
    cand_dic, ent_dic = GenerateCand('kb.json')

# 编码类
data_encoder = DataEncoder(vocab["w2i"], utils.type2label)

# 实例化模型
model = Model(matrix, utils.param)
model.load_state_dict(torch.load('./weight/ckpt_best_2.pth')['net'])

# 匹配输入文本：eg. ********[mention1]********[mention1]****
patten = re.compile(r'\[[^\]]+\]')
text_id = 0

def predict(sentence):
    global text_id
    text_id += 1
    predict_line = {}
    predict_line['text_id'] = str(text_id)
    predict_line['text'] = sentence.replace('[', '').replace(']', '')
    predict_line['mention_data'] = []
    it = re.finditer(patten, sentence)
    for i, match in enumerate(it):
        predict_line['mention_data'].append({
            "mention":
            match.group()[1:-1],
            "offset":
            str(match.start() - 2 * i)
        })
    jsonstr = json.dumps(predict_line, ensure_ascii=False)
    with open("./data/basic_data/predict.json", 'w', encoding='utf-8') as jsonfile:
        jsonfile.write(jsonstr)
    # 生成预测的文本数据
    GeneratePairwaiseSample('predict.json', cand_dic, ent_dic, is_train=False)
    # 数据编码
    data_encoder.data_encode("./data/generated/predict_data.txt",
                             is_train=False)
    # 构建数据集加载接口
    predict_set = DataSet("./data/generated/predict.csv", is_train=False)
    # dataloader
    test_loader = DATA.DataLoader(predict_set,
                                  batch_size=8,
                                  collate_fn=utils.collate_fn_test)
    result = {}
    for i, test_data in enumerate(test_loader):
        id_list, query, offset, cand_desc, seq_len = test_data
        # forward
        pre_label, pre_type = model.predict(query, offset, cand_desc, seq_len)
        # 记录预测结果
        result = utils.record(result, id_list, torch.softmax(pre_label, dim=-1), pre_type)
    # 处理预测结果，生成打印信息
    data = []
    with open('./data/basic_data/predict.json', 'r', encoding='utf8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    for i, line in enumerate(data):
        res_line = result[line['text_id']]
        mention_data = line["mention_data"]
        for mid, item in enumerate(line["mention_data"]):
            item['pre_id'] = res_line[str(mid)]['pre_id']
            pre_type_id = res_line[str(mid)]['pre_type'].argmax().item()
            item['pre_type'] = utils.lable2type[pre_type_id]
            if item['pre_id'] != 'NIL':
                item["pre_desc"] = ent_dic[item['pre_id']]['ent_desc']
                item['pre_type'] = ent_dic[item['pre_id']]['type']
            mention_data[mid] = item
        data[i]['mention_data'] = mention_data
    # 打印结果
    f = open('./data/basic_data/predict_result.json', 'w+', encoding='utf8')
    for i in data:
        print(i['text'])
        for j in i['mention_data']:
            print("实体:\t", j['mention'])
            print("类型:\t", j['pre_type'])
            if j['pre_id'] != 'NIL':
                print('描述:\t', j['pre_desc'])
            print('\n')
        f.write(json.dumps(i, ensure_ascii=False) + '\n')
    f.close()
    return data


if __name__ == "__main__":

    while 1:
        # [《绿皮书》][托尼利普]和[唐博士]，配上这首[歌]，网友：这种[情愫]有点嗲
        sentence = input('>>>  ')
        predict(sentence)
        