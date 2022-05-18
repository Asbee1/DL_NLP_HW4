import re,os
import numpy as np
import jieba
from gensim.models import word2vec

def get_data():
    """如果文档还没分词，就进行分词"""
    outfilename_1 = "./train_jieba.txt"
    if not os.path.exists('./train_jieba.txt'):
        outputs = open(outfilename_1, 'w', encoding='UTF-8')
        datasets_root = "./datasets1"
        catalog = "inf.txt"

        test_num = 10
        test_length = 20
        with open(os.path.join(datasets_root, catalog), "r", encoding='utf-8') as f:
            all_files = f.readline().split(",")
            print(all_files)

        for name in all_files:
            with open(os.path.join(datasets_root, name + ".txt"), "r", encoding='utf-8') as f:
                file_read = f.readlines()
                train_num = len(file_read) - test_num
                choice_index = np.random.choice(len(file_read), test_num + train_num, replace=False)
                train_text = ""
                for train in choice_index[0:train_num]:
                    line = file_read[train]
                    line = re.sub('\s', '', line)
                    line = re.sub('[\u0000-\u4DFF]', '', line)
                    line = re.sub('[\u9FA6-\uFFFF]', '', line)
                    if len(line) == 0:
                        continue
                    seg_list = list(jieba.cut(line, cut_all=False))  # 使用精确模式
                    line_seg = ""
                    for term in seg_list:
                        line_seg += term + " "
                        # for index in range len(line_seg):
                    outputs.write(line_seg.strip() + '\n')
        outputs.close()
        print("得到训练集！！！")



get_data()

fr = open('./train_jieba.txt', 'r', encoding='utf-8')
train = []
for line in fr.readlines():
    line = [word.strip() for word in line.split(' ')]
    train.append(line)

num_features = 300  # Word vector dimensionality
min_word_count = 10  # Minimum word count
num_workers = 16  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words
sentences = word2vec.Text8Corpus("train_jieba.txt")

model = word2vec.Word2Vec(sentences, workers=num_workers,
                            vector_size=num_features, min_count=min_word_count,
                            window=context, sg=1, sample=downsampling)
model.init_sims(replace=True)
# 保存模型
model.save("model_px.model")


