import csv
import json
import jieba
import random
import string
import torch
import torch.utils.data as DATA
from tqdm import tqdm


def GenerateCand(file_path, save=True):
    """预处理知识库，生成候选实体字典、知识库字典、保存所有实体名

    Args:
        file_path (string): 知识库文件名
        save (bool, optional): 是否保存候选实体字典. Defaults to True.

    Returns:
        cand_dic, ent_dic: 候选实体字典、知识库字典
    """
    ent_dic = {}  # 实体字典 {kb_id：<entity, desc, type>}
    cand_dic = {}  # 候选实体集合 {mention: [id_list]}
    for line in open('../data/basic_data/' + file_path, encoding='utf-8'):
        line_json = json.loads(line.strip())
        mention_list = line_json.get('alias')
        subject_name = line_json.get('subject')
        mention_list.append(subject_name)
        subject_id = line_json.get('subject_id')
        subject_type = line_json.get('type')
        ent_desc = ''
        for item in line_json.get('data'):
            ent_desc += item.get('predicate') + ':' + item.get('object') + ';'
        # generate ent_dic
        ent_dic[subject_id] = {}
        ent_dic[subject_id]['ent_name'] = subject_name
        ent_dic[subject_id]['ent_desc'] = ent_desc
        ent_dic[subject_id]['type'] = subject_type
        # generate cand_dic
        for mention in mention_list:
            if mention in cand_dic:
                cand_dic[mention]['iid_list'].append(subject_id)
                cand_dic[mention]['iid_list'] = list(
                    set(cand_dic[mention]['iid_list']))
            else:
                cand_dic[mention] = {}
                cand_dic[mention]['iid_list'] = [subject_id]
    # 保存数据
    if save:
        jsonstr = json.dumps(cand_dic, ensure_ascii=False, indent=4)
        with open("../data/generated/cand.json", 'w', encoding='utf-8') as jsonfile:
            jsonfile.write(jsonstr)
        jsonstr = json.dumps(ent_dic, ensure_ascii=False, indent=4)
        with open("../data/generated/entity.json", 'w', encoding='utf-8') as jsonfile:
            jsonfile.write(jsonstr)
    return cand_dic, ent_dic


def GeneratePairwaiseSample(file_path, cand_dic, ent_dic, is_train=True):
    """生成成对的训练或预测样本（格式如下）
    is_train=True: | text_id | mention_id | query | mention | cand1_desc | cand2_desc | cand1_id | cand2_id | lable | type kb_id
    is_Train=False: | text_id | mention_id | query | mention | cand_desc | cand_id | type | kb_id |

    Args:
        file_path (string): 原始数据
        cand_dic (dict): 候选实体集合
        ent_dic (dict): 知识库实体
        is_train (bool, optional): 是否生成训练集格式. Defaults to True.
    """
    # 保存训练数据
    data_fname = file_path.replace('.json', '_data.txt')
    out_file = open('../data/generated/' + data_fname, 'w', encoding='utf-8')
    # 处理数据，生成查询文本(query)和候选实体对(ent_pair)数据
    for line in open('../data/basic_data/' + file_path, encoding='utf-8'):
        line_json = json.loads(line.strip())
        text_id = line_json.get('text_id')
        query = line_json.get('text')
        mention_data = line_json.get('mention_data')
        mention_id = 0
        for item in mention_data:
            mention, offset = item.get('mention'), item.get('offset')
            kb_id = item.get('kb_id',
                             'None')  # 获取最佳匹配结果, is_train=False 时为 None
            if kb_id == 'None':  # is_train=False
                golden_type = 'None'
            elif 'NIL' in kb_id:
                if '|' in kb_id:
                    kb_id = kb_id.split('|')[0]
                golden_desc = mention
                golden_type = kb_id.replace('NIL_', '')
                kb_id = 'NIL'
            else:
                golden_desc = ent_dic[kb_id]['ent_desc']
                golden_type = ent_dic[kb_id]['type']
                if '|' in golden_type:
                    golden_type = golden_type.split('|')[0]
            # 匹配候选实体集
            if mention in cand_dic:
                cand = cand_dic[mention]
                iid_list = cand['iid_list']
            elif not is_train:
                cand = {}
                iid_list = []
            else:
                continue
            # 将 NIL 添加到每一个 mention 的候选实体集合中
            iid_list.append('NIL')
            iid_list = list(set(iid_list))
            cand['iid_list'] = iid_list
            # 生成 ent_pair 实体对 (pairwise 模型)
            for iid in iid_list:
                if iid != 'NIL':
                    tmp_desc = ent_dic[iid]['ent_desc']
                else:
                    tmp_desc = mention
                if not is_train:
                    out_file.write(text_id + '\t' + str(mention_id) + '\t' +
                                   query + '\t' + mention + '\t' + tmp_desc +
                                   '\t' + iid + '\t' + golden_type + '\t' +
                                   kb_id + '\n')
                    continue
                elif iid == kb_id:
                    continue
                # pair 随机生成正样本（前面的相似度大于后面）和负样本（反之）
                random_threshold = random.random()
                if random_threshold > 0.5:
                    out_file.write(text_id + '\t' + str(mention_id) + '\t' +
                                   query + '\t' + mention + '\t' +
                                   golden_desc + '\t' + tmp_desc + '\t' +
                                   kb_id + '\t' + iid + '\t' + '1' + '\t' +
                                   golden_type + '\t' + kb_id + '\n')
                else:
                    out_file.write(text_id + '\t' + str(mention_id) + '\t' +
                                   query + '\t' + mention + '\t' + tmp_desc +
                                   '\t' + golden_desc + '\t' + iid + '\t' +
                                   kb_id + '\t' + '0' + '\t' + golden_type +
                                   '\t' + kb_id + '\n')
            mention_id += 1
    out_file.close()


class DataEncoder(object):
    """对数据进行编码处理"""
    def __init__(self,
                 word2idx,
                 type2label,
                 user_word_dict="../data/generated/mention.txt"):
        self.word2idx = word2idx
        self.type2label = type2label
        jieba.load_userdict(user_word_dict)

    def tokenize(self, text):
        # jieba 分词
        text = jieba.lcut(text, HMM=True)
        # 去掉标点符号
        return text 

    def data_encode(self, fname, is_train=True):
        data_file = open(fname.replace("_data.txt", ".csv"),
                         'w',
                         encoding='utf-8',
                         newline='')
        writer = csv.writer(data_file)
        with open(fname, 'r', encoding='utf-8') as f:
            if is_train:
                writer.writerow([
                    'text_id', 'mention_id', 'query', 'offset', 'cand1_desc',
                    'cand2_desc', 'cand1_id', 'cand2_id', 'label', 'type',
                    'golden_id'
                ])
                for line in tqdm(f,
                                 desc='Encode ' + fname.split('/')[-1][:-4]):
                    try:
                        line = line.strip().split('\t')
                        query = self.tokenize(line[2])
                        mention = line[3]
                        line[3] = offset = [
                            i for i, x in enumerate(query)
                            if x.find(mention) != -1
                        ][0]
                        line[2] = query = [
                            self.word2idx[word] if word in self.word2idx else
                            self.word2idx['<unk>'] for word in query
                        ]
                        line[4] = cand1_desc = [
                            self.word2idx[word] if word in self.word2idx else
                            self.word2idx['<unk>']
                            for word in self.tokenize(line[4])
                        ]
                        line[5] = cand2_desc = [
                            self.word2idx[word] if word in self.word2idx else
                            self.word2idx['<unk>']
                            for word in self.tokenize(line[5])
                        ]
                        line[9] = type_id = self.type2label[line[9]]
                        writer.writerow(line)
                    except:
                        continue
            else:
                writer.writerow([
                    'text_id', 'mention_id', 'query', 'offset', 'cand_desc',
                    'cand_id', 'type', 'golden_id'
                ])
                for line in tqdm(f, desc='Encode ' + fname):
                    line = line.strip().split('\t')
                    query = self.tokenize(line[2])
                    mention = line[3]
                    mchar = mention[int(len(mention) / 2)]
                    line[3] = offset = [
                        i for i, x in enumerate(query) if x.find(mchar) != -1
                    ][0]
                    line[2] = query = [
                        self.word2idx[word]
                        if word in self.word2idx else self.word2idx['<unk>']
                        for word in query
                    ]
                    line[4] = cand_desc = [
                        self.word2idx[word]
                        if word in self.word2idx else self.word2idx['<unk>']
                        for word in self.tokenize(line[4])
                    ]
                    line[6] = type_id = self.type2label.get(line[6], -1)
                    writer.writerow(line)
        data_file.close()


class DataSet(DATA.Dataset):
    """数据集"""
    def __init__(self, path, is_train=True):
        super(DATA.Dataset, self).__init__()
        with open(path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            head = next(reader)
            self.data = [sample for sample in reader]
        self.size = len(self.data)
        self.type_num = 24
        self.is_train = is_train

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        sample = self.data[item]
        if self.is_train:
            # (text_id, mention_id, cand1_id, cand2_id, golden_type, golden_id)
            id_list = (sample[0], sample[1], sample[6], sample[7], sample[9],
                       sample[10])
            query = torch.tensor(json.loads(sample[2]), dtype=torch.long)
            offset = torch.tensor(int(sample[3]))
            cand1_desc = torch.tensor(json.loads(sample[4]), dtype=torch.long)
            cand2_desc = torch.tensor(json.loads(sample[5]), dtype=torch.long)
            label = torch.tensor(int(sample[8]))
            ent_type = torch.tensor(int(sample[9]))
            return id_list, query, offset, cand1_desc, cand2_desc, label, ent_type
        else:
            # (text_id, mention_id, cand_id, golden_type, golden_id)
            id_list = (sample[0], sample[1], sample[5], sample[6], sample[7])
            query = torch.tensor(json.loads(sample[2]), dtype=torch.long)
            offset = torch.tensor(int(sample[3]))
            cand_desc = torch.tensor(json.loads(sample[4]), dtype=torch.long)
            return id_list, query, offset, cand_desc