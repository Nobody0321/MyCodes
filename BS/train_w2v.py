import json
import re
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


stop_words = r"[-,.!'()$/:#@%]"

def gather_all_sens(train_raw_data_path, test_raw_data_path, sens_file_path):
    f = open(sens_file_path, "w", encoding="utf-8")
    train_data = json.load(open(train_raw_data_path, "r", encoding="utf-8"))
    num = 1
    for each in train_data:
        sen = each['sentence'].lower()
        f.write(remove_stopwords(sen))
        f.write("\n")
        num += 1
        if num % 1000 == 0:
            print(num)
    del train_data
    test_data = json.load(open(test_raw_data_path, "r", encoding="utf-8"))
    for each in test_data:
        sen = each['sentence'].lower()
        f.write(remove_stopwords(sen))
        f.write("\n")
        num += 1
        if num %1000 == 0:
            print(num)
    del test_data
    f.close()

def remove_stopwords(sen):
    sen =  re.sub(stop_words, " ", sen)
    sen = " ".join(sen.split())
    return sen

def train_w2v(corpus_file_path):
    sens = open(corpus_file_path, "r", encoding="utf-8").readlines()
    sens = map(lambda x: x.strip().split(), sens)
    model = Word2Vec(sens, size=100, min_count=2, workers=100, iter=10)
    model.save('./nyt_vec-100')


def vec_to_json(w2v_model_path):
    w2v_json = []
    wv = Word2Vec.load(w2v_model_path, mmap='r').wv
    i = 0
    for each in wv.index2entity:
        w2v_json.append({"word":each, "vec":wv[each].tolist()})
        i += 1
        if i % 1000 == 0:
            print(i)
    json.dump(w2v_json, open("./vec_100.json", "w", encoding="utf-8"))


if __name__ == "__main__":
    # gather_all_sens("../../raw_data/train.json", "../../raw_data/test.json", "./sens.txt")
    train_w2v("./sens.txt")
    vec_to_json("./nyt_vec-100")
    
