import os
import string
import re

punc = string.punctuation + u"！？＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."


def pureSen(sentence):
    return [char for char in sentence if char not in punc]


def getSens(filename):
    sentences = open(filename, 'r', encoding='utf-8').readline().strip()
    patterns = r'([0-9]{1,2}\.)\D'
    for pattern in re.findall(patterns, sentences):
        sentences = sentences.replace(pattern, '。')
    sentences = sentences.split('。')

    sentences = [pureSen(sen) for sen in sentences if len(pureSen(sen))]
    return sentences


def cut(rawDirs):
    for rawDir in rawDirs:
        filenames = os.listdir(rawDir)
        originalFiles = [os.path.join(rawDir, filename) for filename in filenames if 'original' in filename]
        for filename in originalFiles:
            results = getSens(filename)
            with open(corpusFile, 'a+') as f:
                for line in results:
                    f.write(' '.join(line)+'\n')


def train_W2V(corpus, modelSave):
    from gensim.models import word2vec
    num_features = 300    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 16       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    sentences = open(corpus).readlines()
    sentences = [line.strip().split(' ') for line in sentences]
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                              size=num_features, min_count = min_word_count, \
                              window=context, sg = 1, sample = downsampling)
    model.init_sims(replace=True)
    model.save(modelSave)


if __name__ == "__main__":
    corpusFile = r"./data/sentences.txt"
    modelSave = "./data/sentences.model"
    rawDirs = [r"./data/病史特点", r"./data/一般项目", r"./data/诊疗经过", r"./data/出院情况"]

    cut(rawDirs)
    train_W2V(corpusFile, modelSave)
