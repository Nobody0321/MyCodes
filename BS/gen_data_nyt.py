import numpy as np
import os
import json

in_path = "./raw_data/"
out_path = "./data"
case_sensitive = False
if not os.path.exists(out_path):
	os.mkdir(out_path)
train_file_name = in_path + 'train.json'
test_file_name = in_path + 'test.json'
word_file_name = in_path + 'word_vec.json'
rel_file_name = in_path + 'rel2id.json'


def find_pos(sentence, head, tail):
	"""find each word's pos to head and tail entity
	
	Args:
		sentence (str): sen str
		head (str): head entity str
		tail (str): tail entity str	
	Returns:
		[type]: [description]
	"""
	def find(sentence, entity):
		"""
		find entity index
		"""
		# entity starting pos
		p = sentence.find(' ' + entity + ' ')
		if p == -1:
			# entity part not found
			if sentence[:len(entity) + 1] == entity + ' ':
				# entity at start
				p = 0
			elif sentence[-len(entity) - 1:] == ' ' + entity:
				# entity at end
				p = len(sentence) - len(entity)
			else:
				p = 0
		else:
			p += 1
		return p

	sentence = ' '.join(sentence.split())
	p1 = find(sentence, head)
	p2 = find(sentence, tail)
	words = sentence.split()
	cur_pos = 0
	head_entity_pos = -1
	tail_entity_pos = -1
	for i, word in enumerate(words):
		if cur_pos == p1:
			head_entity_pos = i
		if cur_pos == p2:
			tail_entity_pos = i
		cur_pos += len(word) + 1
	return head_entity_pos, tail_entity_pos


def init(file_name, word_vec_file_name, rel2id_file_name, max_length=120, case_sensitive=False, is_training=True):
	if file_name is None or not os.path.isfile(file_name):
		raise Exception("[ERROR] Data file doesn't exist")
	if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
		raise Exception("[ERROR] Word vector file doesn't exist")
	if rel2id_file_name is None or not os.path.isfile(rel2id_file_name):
		raise Exception("[ERROR] rel2id file doesn't exist")

	print("Loading data file...")
	ori_data = json.load(open(file_name, "r"))
	print("Finish loading")
	print("Loading word_vec file...")
	ori_word_vec = json.load(open(word_vec_file_name, "r"))
	print("Finish loading")
	print("Loading rel2id file...")
	rel2id = json.load(open(rel2id_file_name, "r"))
	print("Finish loading")

	if not case_sensitive:
		print("Eliminating case sensitive problem...")
		for i in ori_data:
			i['sentence'] = i['sentence'].lower()
			i['head']['word'] = i['head']['word'].lower()
			i['tail']['word'] = i['tail']['word'].lower()
		for i in ori_word_vec:
			i['word'] = i['word'].lower()
		print("Finish eliminating")

	# vec
	print("Building word vector matrix and mapping...")
	word2id = {}
	word_vec_mat = []
	word_vec_dim = len(ori_word_vec[0]['vec'])
	print("Got {} words of {} dims".format(len(ori_word_vec), word_vec_dim))
	for i in ori_word_vec:
		word2id[i['word']] = len(word2id)
		word_vec_mat.append(i['vec'])
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	# UNK vector
	word_vec_mat.append(np.random.normal(loc=0, scale=0.05, size=word_vec_dim))
	# blank vector
	word_vec_mat.append(np.zeros(word_vec_dim, dtype=np.float32))
	word_vec_mat = np.array(word_vec_mat, dtype=np.float32)
	print("Finish building")

	# sorting, by head entity then tail entity then relation
	print("Sorting data...")
	ori_data.sort(key=lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
	print("Finish sorting")

	n_sen = len(ori_data)
	sen_word = np.zeros((n_sen, max_length), dtype=np.int64)
	sen_pos0 = np.zeros((n_sen, max_length), dtype=np.int64)
	sen_head_entity_pos = np.zeros((n_sen, max_length), dtype=np.int64)
	sen_tail_entity_pos = np.zeros((n_sen, max_length), dtype=np.int64)
	sen_mask = np.zeros((n_sen, max_length, 3), dtype=np.float32)
	sen_label = np.zeros((n_sen), dtype=np.int64)
	sen_len = np.zeros((n_sen), dtype=np.int64)
	bag_label = []
	bag_scope = []
	bag_key = []
	for i in range(len(ori_data)):
		if i % 1000 == 0:
			print(i)
		data = ori_data[i]
		# sen_label 
		if data['relation'] in rel2id:
			sen_label[i] = rel2id[data['relation']]
		else:
			sen_label[i] = rel2id['NA']

		words = data['sentence'].split()

		# sen_len
		sen_len[i] = min(len(words), max_length)

		# sen_word
		for j, word in enumerate(words):
			if j < max_length:
				if word in word2id:
					sen_word[i][j] = word2id[word]
				else:
					sen_word[i][j] = word2id['UNK']
		# padding
		for j in range(j + 1, max_length):
			sen_word[i][j] = word2id['BLANK']

		head_entity_pos, tail_entity_pos = find_pos(data['sentence'], data['head']['word'], data['tail']['word'])
		if head_entity_pos == -1 or tail_entity_pos == -1:
			raise Exception(
				"[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, data['sentence'],
				                                                                                 data['head']['word'],
				                                                                                 data['tail']['word']))
		if head_entity_pos >= max_length:
			head_entity_pos = max_length - 1
		if tail_entity_pos >= max_length:
			tail_entity_pos = max_length - 1
		pos_min, pos_max = list(sorted([head_entity_pos, tail_entity_pos]))
		for j in range(max_length):
			# sen_head_entity_pos, sen_tail_entity_pos
			sen_pos0[i][j] = j
			sen_head_entity_pos[i][j] = j - head_entity_pos + max_length
			sen_tail_entity_pos[i][j] = j - tail_entity_pos + max_length
			# sen_mask
			if j >= sen_len[i]:
				sen_mask[i][j] = [0, 0, 0]
			elif j - pos_min <= 0:
				sen_mask[i][j] = [100, 0, 0]
			elif j - pos_max <= 0:
				sen_mask[i][j] = [0, 100, 0]
			else:
				sen_mask[i][j] = [0, 0, 100]
		# bag_range	
		if is_training:
			# sort train data by entity pair and relation
			tup = (data['head']['id'], data['tail']['id'], data['relation'])
		else:
			# sort test data by entity pair
			tup = (data['head']['id'], data['tail']['id'])
		if bag_key == [] or bag_key[len(bag_key) - 1] != tup:
			bag_key.append(tup)
			bag_scope.append([i, i])
		bag_scope[len(bag_scope) - 1][1] = i

	print("Processing bag label...")
	# bag_label
	if is_training:
		for i in bag_scope:
			bag_label.append(sen_label[i[0]])
	else:
		for i in bag_scope:
			# for testing data, each bag has a multi-hot label
			multi_hot = np.zeros(len(rel2id), dtype=np.int64)
			for j in range(i[0], i[1] + 1):
				multi_hot[sen_label[j]] = 1
			bag_label.append(multi_hot)
	print("Finish processing")

	# ins_scope
	ins_scope = np.stack([list(range(len(ori_data))), list(range(len(ori_data)))], axis=1)
	print("Processing instance label...")
	# ins_label
	if is_training:
		ins_label = sen_label
	else:
		ins_label = []
		for i in sen_label:
			one_hot = np.zeros(len(rel2id), dtype=np.int64)
			one_hot[i] = 1
			ins_label.append(one_hot)
		ins_label = np.array(ins_label, dtype=np.int64)
	print("Finishing processing")
	bag_scope = np.array(bag_scope, dtype=np.int64)
	bag_label = np.array(bag_label, dtype=np.int64)
	ins_scope = np.array(ins_scope, dtype=np.int64)
	ins_label = np.array(ins_label, dtype=np.int64)

	# saving
	print("Saving files")
	if is_training:
		name_prefix = "train"
	else:
		name_prefix = "test"
		
	np.save(os.path.join(out_path, 'vec.npy'), word_vec_mat)
	np.save(os.path.join(out_path, name_prefix + '_word.npy'), sen_word)
	np.save(os.path.join(out_path, name_prefix + '_pos0.npy'), sen_pos0)
	np.save(os.path.join(out_path, name_prefix + '_pos1.npy'), sen_head_entity_pos)
	np.save(os.path.join(out_path, name_prefix + '_pos2.npy'), sen_tail_entity_pos)
	np.save(os.path.join(out_path, name_prefix + '_mask.npy'), sen_mask)
	np.save(os.path.join(out_path, name_prefix + '_bag_label.npy'), bag_label)
	np.save(os.path.join(out_path, name_prefix + '_bag_scope.npy'), bag_scope)
	np.save(os.path.join(out_path, name_prefix + '_ins_label.npy'), ins_label)
	np.save(os.path.join(out_path, name_prefix + '_ins_scope.npy'), ins_scope)
	print("Finish saving")


init(train_file_name, word_file_name, rel_file_name, max_length=120, case_sensitive=False, is_training=True)
init(test_file_name, word_file_name, rel_file_name, max_length=120, case_sensitive=False, is_training=False)
