<h1 style="text-align: center"> 基于实体链接的实体消歧系统 </h1>

## 1、文件结构树

```shell
.
│  .gitignore
│  demo.ipynb
│  LICENSE
│  main.py
│  README.md
│  report.ipynb
│  requirements.txt
│
├─data
│  │  eval.py
│  │
│  ├─basic_data
│  │      CCKS 2020 Entity Linking License.docx
│  │      dev.json
│  │      kb.json
│  │      README
│  │
│  ├─generated
│  │
│  └─pretrain_data
│          matrix.npy
│          vocab.json
│          word2vec.iter5
│
├─img
│      model.png
│      word2vec.png
│
├─results
│      result_test_example.json
│
├─src
│      data_process.py
│      model.py
│      train.py
│      utils.py
│      __init__.py
│
└─weights
      ckpt_best_0.pth
      ckpt_best_1.pth
      ckpt_best_2.pth
      placeholder.txt
```



## 2、代码说明

***[report.ipynb](/report.ipynb)***

包含模型的详细描述、数据结构、中间数据，以及数据处理的完整过程。

***[data_process.py](/src/data_process.py)***

* `DataEncoder`：用于将文本训练数据进行编码，将文本序列分词、规范化、编码，对标签进行编码

* `DataSet`：继承自 ***torch.utils.data.Dataset*** 数据加载类

* `GenerateCand`（函数）

  > 预处理知识库，生成候选实体字典、知识库字典、保存所有实体名。
  > 处理 ***kb.json*** 文件得到 ***cand.json*** 、***entity.json*** 、***mention.txt***

* `GeneratePairwaiseSample` （函数）

  >  消歧系统采用 pairwise 学习排序算法，对候选实体的排序。
  >  * 此函数默认参数 `is_train=True` 根据训练集生成文本形式的训练数据。对于一个实体提及，生成两个候选实体，cand1、cand2，其中正样本 (label=1) 默认 cand1 是最优匹配实体，负样本反之。
  >  * 针对测试和验证数据，不使用pairwise，从候选实体集中给该提及 (mention) 一个候选实体，输入模型打分。
  

***[model.py](/src/model.py)***

包含 `Model` 类，即实现候选实体排序、实体类型预测的多任务模型



***[utils.py](/src/utils.py)***

* `param`： 包含参数字典 ；
* `type2label`： 类型到标签的映射字典 ；
* `label2type`： 标签到类型的映射列表 ；
* `loadWord2Vec` ：词向量加载函数；
* `collate_fn_train`、`collate_fn_test`：dataloader 的 batch 数据处理函数 ；
* `record` ：记录预测结果函数 ；
* `Accuracy`：计算 Accuracy 的函数



## 3、使用说明

1. 配置 python 环境

    ```shell
    # 使用 pip 安装程序运行环境
    $ pip install -r requirements.txt
    ```

2. 下载数据和预训练的模型权重：

   * 下载 data.zip，解压后，放入项目文件夹下：

      百度网盘链接：<https://pan.baidu.com/s/1W-m-wqeU-DX6AqJ69OWL2Q>
      提取码：etoz 

   * 下载 weight 文件夹，解压后，放入项目文件夹下 

      百度网盘链接：<https://pan.baidu.com/s/1uVP_Jd8OYpV5t5M98f2tqA>
      提取码：20au 


3. 在终端打开项目文件夹，调用模型脚本 [main.py](./main.py) 进入交互接口

   > 在交互式接口中，输入文本，将待消歧的词语用 `[]` 分隔，即可得到输出

   ```bash
   # 调用接口脚本
   $ python .\main.py
   Building prefix dict from the default dictionary ...
   Loading model from cache C:\Users\xuzf\AppData\Local\Temp\jieba.cache
   Loading model cost 1.712 seconds.
   Prefix dict has been built succesfully.
   >>>[《绿皮书》][托尼·利普]和[唐博士]，配上这首[歌]，网友：这种[情愫]有点嗲
   Encode ./data/generated/predict_data.txt: 28it [00:00, 241.86it/s]
   《绿皮书》托尼·利普和唐博士，配上这首歌，网友：这种情愫有点嗲
   实体:    《绿皮书》
   类型:    Work
   描述:    外文名:Green Book;摘要:《绿皮书》是由彼得·法拉利执导，维果·莫特森、马赫沙拉·阿里主演的剧情片，于2018年9月11日在多伦多国际电
   影节首映；2019年3月1日在中国内地上映。;制片地区:美国;编剧:Brian Currie、彼得·法拉利、尼克·维勒欧嘉;片长:130分钟;对白语言:英语;主演:维果·莫特森，马赫沙拉·阿里;导演:彼得·法拉利;中文名:绿皮书;发行公司:环球影业;上映时间:2018年9月11日(多伦多国际电影节)、2019年3月1日（中
   国内地）;类型:剧情片;义项描述:美国2018年彼得·法拉利执导电影;主要奖项:第91届奥斯卡金像奖最佳影片;色彩:彩色;其它译名:绿书、绿簿旅友、幸福绿皮书;
   
   
   实体:    托尼利普
   类型:    Other
   描述:    出生地:美国宾夕法尼亚州比弗福尔斯;外文名:Tony Lip;摘要:托尼·利普（Tony Lip，原名Frank Anthony Vallelonga，1930年7月30日-2013年1月4日），是电影《绿皮书》中白人司机的原型。;逝世日期:2013年1月4日;别名:Frank Anthony Vallelonga;义项描述:托尼·利普;中文名:托尼·利普;国籍:美国;出生日期:1930年7月30日;
   
   
   实体:    唐博士
   类型:    Other
   
   
   实体:    歌
   类型:    Other
   描述:    外文名:song;摘要:歌，读音gē，哥，对唱情歌；欠，感叹。;拼音:gē;注音:ㄍㄜˉ;五笔:SKSW;中文名:歌;义项描述:汉语汉字;类别:汉语汉字;标签:音乐、字词;
   
   
   实体:    情愫
   类型:    Work
   描述:    摘要:《情愫》是2011年云南人民出版社出版出版的图书，作者是陈强。;出版社:云南人民出版社出版;作者:陈强;义项描述:陈强诗集;出版时间:2011年11月;书名:情愫;标签:艺术书籍、出版物、书籍;
   
   
   >>>
   ```

4.  `ctrl + c` 结束命令行交互