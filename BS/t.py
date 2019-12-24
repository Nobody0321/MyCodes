import json
in_path = "./raw_data/"

train_file_name = in_path + 'train.json'

ori_data = json.load(open(train_file_name, "r"))

max_l = 0
min_l = 100000
for i in range(len(ori_data)):
    l = len(ori_data[i]["sentence"])
    print(l)
    if l > max_l:
        max_l = l
    if l < min_l:
        min_l = l

print(max_l)
print(min_l)