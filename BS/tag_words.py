import numpy as np
import os
import json


def find_pos(sentence, head, tail):
    """
    找到头尾实体在句子中的位置（头尾实体可能是单词或者词组）
    """

    def find_entity(sentence, entity):
        p = sentence.find(" " + entity + " ")
        if p == -1:
            if sentence[:len(entity) + 1] == entity + " ":
                p = 0
            elif sentence[-len(entity) - 1:] == " " + entity:
                p = len(sentence) - len(entity)
            else:
                p = 0
        else:
            p += 1
        return p

    sentence = " ".join(sentence.split())
    p1 = find_entity(sentence, head)
    p2 = find_entity(sentence, tail)
    words = sentence.split()
    cur_pos = 0
    pos1 = -1
    pos2 = -1
    for i, word in enumerate(words):
        if cur_pos == p1:
            pos1 = i
        if cur_pos == p2:
            pos2 = i
        cur_pos += len(word) + 1
    return pos1, pos2


def init(file_name, word_vec_file_name, rel2id_file_name, sen_max_length=120,
         case_sensitive=False, is_training=True, small_bag=None):
    if file_name is None or not os.path.isfile(file_name):
        raise Exception("[ERROR] Data file does not exist")
    if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
        raise Exception("[ERROR] Word vector file does not exist")
    if rel2id_file_name is None or not os.path.isfile(rel2id_file_name):
        raise Exception("[ERROR] rel2id file does not exist")

    print("Loading data file...")
    # ori_data:{"head":{"type":head entity type,"word":head entity word, "id":head entity id in kb}
    #           "tail":{"type":"","word":"", id}
    #           "sentence":the sentence}
    ori_data = json.load(open(file_name, "r"))
    print("Finish loading")
    print("Loading word_vec file...")
    # ori_word_vec:{"word":word_str, "vec":word_vec}
    ori_word_vec = json.load(open(word_vec_file_name, "r"))
    print("Finish loading")
    print("Loading rel2id file...")
    # rel2id: {relation_str: id}
    rel2id = json.load(open(rel2id_file_name, "r"))
    print("Finish loading")

    if not case_sensitive:
        print("Eliminating case sensitive problem...")
        for i in ori_data:
            i["sentence"] = i["sentence"].lower()
            i["head"]["word"] = i["head"]["word"].lower()
            i["tail"]["word"] = i["tail"]["word"].lower()
        for i in ori_word_vec:
            # convert all words in word2vec file to lower case
            i["word"] = i["word"].lower()
        print("Finish eliminating")

    print("Start building word vector matrix and mapping...")
    word2id = {}
    word_vec_matrix = []  # stores all word2vec vector
    word_vec_size = len(ori_word_vec[0]["vec"])
    print(("Got {} words of {} dims".format(len(ori_word_vec), word_vec_size)))
    for i in ori_word_vec:
        # mapping word to id
        word2id[i["word"]] = len(word2id)
        word_vec_matrix.append(i["vec"])

    # add unknown and blank tag to wordvec and word2id map
    word2id["UNK"] = len(word2id)
    word2id["BLANK"] = len(word2id)
    # use a normal distribution to represent Unknown label
    word_vec_matrix.append(np.random.normal(loc=0, scale=0.05, size=word_vec_size))
    # use zeros to represent Blank label
    word_vec_matrix.append(np.zeros(word_vec_size, dtype=np.float32))
    # to np
    word_vec_matrix = np.array(word_vec_matrix, dtype=np.float32)
    print("Finish building")

    # sort corpus by head entity id, tail id then relation id
    print("Sorting Corpus...")
    # here we sort all sentences so we can
    ori_data.sort(key=lambda a: a["head"]["id"] + "#" + a["tail"]["id"] + "#" + a["relation"])
    print("Finish sorting")

    sen_num = len(ori_data)
    sen_word = np.zeros((sen_num, sen_max_length), dtype=np.int64)  # convert each word in sentence to word id
    sen_pos1 = np.zeros((sen_num, sen_max_length), dtype=np.int64)
    sen_pos2 = np.zeros((sen_num, sen_max_length), dtype=np.int64)
    # sen_mask = np.zeros((sen_num, sen_max_length, 3), dtype = np.float32)
    senid_2_relationid = np.zeros(sen_num, dtype=np.int64)
    sen_len = np.zeros(sen_num, dtype=np.int64)  # 保存每个句子的长度
    bag_label = []
    bag_scope = []  # used to record [i:j] sentences are the same label
    bag_key = []

    for i in range(len(ori_data)):
        if i % 1000 == 0:
            print(i)
        data = ori_data[i]
        # sen_label
        if data["relation"] in rel2id:
            # 如果这句话的relation 在rel2id词典中，就保存relation id
            senid_2_relationid[i] = rel2id[data["relation"]]
        else:
            senid_2_relationid[i] = rel2id["NA"]
        words = data["sentence"].split()

        # each sen len
        sen_len[i] = min(len(words), sen_max_length)
        # sen_word
        for j, word in enumerate(words):
            if j < sen_max_length:
                if word in word2id:
                    sen_word[i][j] = word2id[word]
                else:
                    sen_word[i][j] = word2id["UNK"]

        for j in range(j + 1, sen_max_length):
            # 长度不足部分pad为blank word
            sen_word[i][j] = word2id["BLANK"]

        pos1, pos2 = find_pos(data["sentence"], data["head"]["word"], data["tail"]["word"])
        if pos1 == -1 or pos2 == -1:
            raise Exception(
                "[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, data["sentence"],
                                                                                                 data["head"]["word"],
                                                                                                 data["tail"]["word"]))
        if pos1 >= sen_max_length:
            pos1 = sen_max_length - 1
        if pos2 >= sen_max_length:
            pos2 = sen_max_length - 1
        # pos_min = min(pos1, pos2)
        # pos_max = max(pos1, pos2)
        for j in range(sen_max_length):
            # sen_pos1, sen_pos2
            sen_pos1[i][j] = j - pos1 + sen_max_length
            sen_pos2[i][j] = j - pos2 + sen_max_length
        # # sen_mask
        # if j >= sen_len[i]:
        # 	sen_mask[i][j] = [0, 0, 0]
        # elif j - pos_min <= 0:
        # 	sen_mask[i][j] = [100, 0, 0]
        # elif j - pos_max <= 0:
        # 	sen_mask[i][j] = [0, 100, 0]
        # else:
        # 	sen_mask[i][j] = [0, 0, 100]
        # bag_scope
        if is_training:
            tup = (data["head"]["id"], data["tail"]["id"], data["relation"])
        else:
            # test data，只考虑是不是一个实体对
            tup = (data["head"]["id"], data["tail"]["id"])
        # 三元组/二元组有变，开辟新的scope
        if bag_key == [] or bag_key[-1] != tup or (small_bag and i - bag_scope[len(bag_scope) - 1][1] > small_bag):
            bag_key.append(tup)
            bag_scope.append([i, i])
        else:  # 不是新的scope，更新之前的scope
            bag_scope[len(bag_scope) - 1][1] = i

    print("Processing bag label...")
    # bag_label: stores label id for each bag in training,
    # and label id for each sentence in a bag, multi hot,
    # when testing, we hope the model to predict all relations in a bag
    if is_training:
        # fot training data, a bag of sentences have the same entity pair
        for each_scope in bag_scope:
            bag_label.append(senid_2_relationid[each_scope[0]])
    else:
        # for testing data, a bag consists of multi label sentences
        for each_scope in bag_scope:
            multi_hot = np.zeros(len(rel2id), dtype=np.int64)
            for j in range(each_scope[0], each_scope[1] + 1):
                multi_hot[senid_2_relationid[j]] = 1
            bag_label.append(multi_hot)
    print("Finish processing")

    # ins_scope
    ins_scope = np.stack([list(range(sen_num)), list(range(sen_num))], axis=1)  # (n, n)
    print("Processing instance label...")
    # sentence label
    if is_training:
        # for training data
        ins_label = senid_2_relationid  # assign real label to training sentences
    else:
        ins_label = []
        # format one-hot label vec for test sentences
        for i in senid_2_relationid:
            one_hot = np.zeros(len(rel2id), dtype=np.int64)
            one_hot[i] = 1
            ins_label.append(one_hot)
        ins_label = np.array(ins_label, dtype=np.int64)
    print("Finishing processing")
    bag_scope = np.array(bag_scope, dtype=np.int64)
    bag_label = np.array(bag_label, dtype=np.int64)  # (bag_num)
    ins_scope = np.array(ins_scope, dtype=np.int64)  # (n, n)
    ins_label = np.array(ins_label, dtype=np.int64)

    # saving
    print("Saving files")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "test"
    np.save(os.path.join(out_path, "vec.npy"), word_vec_matrix)
    np.save(os.path.join(out_path, name_prefix + "_word.npy"), sen_word)
    np.save(os.path.join(out_path, name_prefix + "_pos1.npy"), sen_pos1)
    np.save(os.path.join(out_path, name_prefix + "_pos2.npy"), sen_pos2)
    # np.save(os.path.join(out_path, name_prefix + "_mask.npy"), sen_mask)
    np.save(os.path.join(out_path, name_prefix + "_bag_label.npy"), bag_label)
    np.save(os.path.join(out_path, name_prefix + "_bag_scope.npy"), bag_scope)
    np.save(os.path.join(out_path, name_prefix + "_ins_label.npy"), ins_label)
    np.save(os.path.join(out_path, name_prefix + "_ins_scope.npy"), ins_scope)
    print("Finish saving")


if __name__ == "__main__":

    in_path = "./raw_data/"
    out_path = "data_smallbag"
    case_sensitive = False
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    train_file_name = in_path + "train.json"
    test_file_name = in_path + "test.json"
    word_file_name = in_path + "word_vec.json"
    rel_file_name = in_path + "rel2id.json"

    init(train_file_name, word_file_name, rel_file_name, sen_max_length=230, case_sensitive=False, is_training=True, small_bag=20)
    init(test_file_name, word_file_name, rel_file_name, sen_max_length=230, case_sensitive=False, is_training=False, small_bag=20)
