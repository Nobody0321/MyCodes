import json
import nltk
import test
# tokens = nltk.word_tokenize(sentence)
# pos_tags = nltk.pos_tag(tokens)
in_path = "./raw_data/"
case_sensitive = False
test_file_name = in_path + 'test.json'
word_file_name = in_path + 'word_vec.json'
rel_file_name = in_path + 'rel2id.json'
ori_data = json.load(open(test_file_name, "r"))
ori_data.sort(key = lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])

sentence = ori_data[0]['sentence'].lower()

test.test_nlpir(sentence)