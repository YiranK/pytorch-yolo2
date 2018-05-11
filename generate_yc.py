import os
import json
import numpy as np

idx_f = open('/mnt/lustre/kangyiran2/zero-shot-detection/dataset/indofClsAndAttrCompressed.json', 'r')
data = json.load(idx_f)
for key, value in data.items():
    print (key)
seen_cls = data['seen_class']
unseen_cls = data['unseen_class']
attributes = data['attribute']
num_cls = len(seen_cls)
num_attr = len(attributes)
print (num_cls, num_attr)

yc = np.zeros((num_cls, num_attr))

with open('/mnt/lustre/kangyiran2/zero-shot-detection/cls_attributes.txt', 'r') as f:
    for line in f.readlines():
        cls, attr_list = line.strip().split(':')
        if cls in seen_cls:
            cls_idx = seen_cls.index(cls)
        else:
            if cls in unseen_cls:
                print (cls, 'belongs to unseen_cls')
                continue
        for attr in attr_list.split(','):
            # print (attr)
            attr_idx = attributes.index(attr)
            yc[cls_idx, attr_idx] = 1

np.savetxt('./yc.txt', yc, fmt='%i')


